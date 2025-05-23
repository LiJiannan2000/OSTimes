import torch

from torch.utils.data.dataloader import DataLoader
from utils.utils import AverageMeter
from core.loss import cosine_similarity_loss, order_constraint_loss


def concordance_index(risk, ostime, death):
    # ground truth matrix
    gt_mat = torch.triu(torch.ones(len(ostime), len(ostime))).cuda(device=risk.device)
    gt_mat = gt_mat - torch.diag_embed(torch.diag(gt_mat))
    # pred matrix
    pred_mat = torch.zeros_like(gt_mat)
    for i in range(len(risk)):
        for j in range(len(risk)):
            if death[i] == 0:
                gt_mat[i, j] = 0
                pred_mat[i, j] = 0
            else:
                if risk[i] > risk[j]:
                    pred_mat[i, j] = 1
                elif risk[i] == risk[j]:
                    pred_mat[i, j] = 1 if ostime[i] == ostime[j] else 0.5
                else:
                    pred_mat[i, j] = 0
    # c_index
    if torch.sum(gt_mat) == 0:
        return torch.sum(gt_mat)
    c_index = torch.sum(pred_mat * gt_mat) / torch.sum(gt_mat)

    return c_index


def train(model, train_dataset, optimizer, criterion_cox, logger, config, epoch, writer):
    model.train()
    cox_losses = AverageMeter()
    cosine_losses = AverageMeter()
    listnet_losses = AverageMeter()
    losses = AverageMeter()
    scaler = torch.cuda.amp.GradScaler()
    loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )
    print("train data size:", len(train_dataset))
    risks, ostimes, deaths = [], [], []
    for idx, data_dict in enumerate(loader):
        data = data_dict['data']
        data_def = data_dict['data_def']
        death = data_dict['death']
        ostime = data_dict['ostime']
        info = data_dict['info']

        masks = torch.zeros(4, dtype=torch.int)
        masks[1:] = torch.randint(0, 2, (3,))
        masks = masks.repeat(len(config.TRAIN.DEVICES), 1)

        data = data.cuda()
        data_def = data_def.cuda()
        death = death.cuda()
        ostime = ostime.cuda()
        info = info.cuda()
        masks = masks.cuda()
        # run training
        with torch.cuda.amp.autocast(dtype=torch.float32):
            risk, fea_ori, fea_pre = model(data, data_def, info, masks)
            with torch.no_grad():
                ostime, indices = torch.sort(ostime)
                death = death[indices]
                fea_ori = fea_ori[indices]
                fea_pre = fea_pre[indices]
                risk = risk[indices]
            loss_cox = criterion_cox(risk[:, -1], ostime, death)
            loss_cosine = cosine_similarity_loss(fea_pre, fea_ori)
            loss_listnet = order_constraint_loss(risk, temperature=1.0)
            loss = loss_cox + (loss_cosine + loss_listnet)
        risks.append(risk[:, -1])
        ostimes.append(ostime)
        deaths.append(death)
        cox_losses.update(loss_cox.item(), config.TRAIN.BATCH_SIZE)
        cosine_losses.update(loss_cosine.item(), config.TRAIN.BATCH_SIZE)
        listnet_losses.update(loss_listnet.item(), config.TRAIN.BATCH_SIZE)
        losses.update(loss.item(), config.TRAIN.BATCH_SIZE)
        # do back-propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        scaler.step(optimizer)
        scaler.update()

        if idx % config.PRINT_FREQ == 0:  # 每5小轮打印一次
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Cox Loss: {cox_loss.val:.3f} ({cox_loss.avg:.3f})\t' \
                  'Cosine Loss: {cosine_loss.val:.3f} ({cosine_loss.avg:.3f})\t' \
                  'ListNet Loss: {listnet_loss.val:.3f} ({listnet_loss.avg:.3f})\t' \
                  'Loss: {loss.val:.3f} ({loss.avg:.3f})'.format(epoch, idx, len(loader), cox_loss=cox_losses,
                                                                 cosine_loss=cosine_losses, listnet_loss=listnet_losses, loss=losses)
            logger.info(msg)
    if risks[-1].shape:
        risks = torch.cat(risks)
        ostimes = torch.cat(ostimes)
        deaths = torch.cat(deaths)
    else:
        risks = torch.cat(risks[:-1])
        ostimes = torch.cat(ostimes[:-1])
        deaths = torch.cat(deaths[:-1])
    ostimes, indices = torch.sort(ostimes)
    risks = risks[indices]
    deaths = deaths[indices]
    c_index = concordance_index(risks, ostimes, deaths).item()
    logger.info(f'Concordance-index: {c_index}')
    writer.add_scalar(tag="cox_loss/train", scalar_value=cox_losses.avg, global_step=epoch)
    writer.add_scalar(tag="cosine_loss/train", scalar_value=cosine_losses.avg, global_step=epoch)
    writer.add_scalar(tag="listnet_loss/train", scalar_value=listnet_losses.avg, global_step=epoch)
    writer.add_scalar(tag="loss/train", scalar_value=losses.avg, global_step=epoch)
    writer.add_scalar(tag="c_index/train", scalar_value=c_index, global_step=epoch)


def inference(model, valid_dataset, criterion_cox, logger, config, best_perf, epoch, writer):
    model.eval()

    cox_losses = AverageMeter()
    cosine_losses = AverageMeter()
    listnet_losses = AverageMeter()
    losses = AverageMeter()
    loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )
    risks, ostimes, deaths = [], [], []
    for idx, data_dict in enumerate(loader):
        data = data_dict['data']
        data_def = data_dict['data_def']
        death = data_dict['death']
        ostime = data_dict['ostime']
        info = data_dict['info']

        masks = torch.zeros(4, dtype=torch.int)
        masks[1:] = torch.randint(0, 2, (3,))
        masks = masks.repeat(len(config.TRAIN.DEVICES), 1)

        data = data.cuda()
        data_def = data_def.cuda()
        death = death.cuda()
        ostime = ostime.cuda()
        info = info.cuda()
        masks = masks.cuda()
        # run training
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float32):
                risk, fea_ori, fea_pre = model(data, data_def, info, masks)
                with torch.no_grad():
                    ostime, indices = torch.sort(ostime)
                    death = death[indices]
                    fea_ori = fea_ori[indices]
                    fea_pre = fea_pre[indices]
                    risk = risk[indices]
                loss_cox = criterion_cox(risk[:, -1], ostime, death)
                loss_cosine = cosine_similarity_loss(fea_pre, fea_ori)
                loss_listnet = order_constraint_loss(risk, temperature=1.0)
                loss = loss_cox + (loss_cosine + loss_listnet)
                risks.append(risk[:, -1])
                ostimes.append(ostime)
                deaths.append(death)
                cox_losses.update(loss_cox.item(), config.TRAIN.BATCH_SIZE)
                cosine_losses.update(loss_cosine.item(), config.TRAIN.BATCH_SIZE)
                listnet_losses.update(loss_listnet.item(), config.TRAIN.BATCH_SIZE)
                losses.update(loss.item(), config.TRAIN.BATCH_SIZE)
    if risks[-1].shape:
        risks = torch.cat(risks)
        ostimes = torch.cat(ostimes)
        deaths = torch.cat(deaths)
    else:
        risks = torch.cat(risks[:-1])
        ostimes = torch.cat(ostimes[:-1])
        deaths = torch.cat(deaths[:-1])

    ostimes, indices = torch.sort(ostimes)
    risks = risks[indices]
    deaths = deaths[indices]

    c_index = concordance_index(risks, ostimes, deaths).item()
    if c_index > best_perf:
        best_perf = c_index

    writer.add_scalar(tag="cox_loss/train", scalar_value=cox_losses.avg, global_step=epoch)
    writer.add_scalar(tag="cosine_loss/train", scalar_value=cosine_losses.avg, global_step=epoch)
    writer.add_scalar(tag="listnet_loss/train", scalar_value=listnet_losses.avg, global_step=epoch)
    writer.add_scalar(tag="loss/val", scalar_value=losses.avg, global_step=epoch)
    writer.add_scalar(tag="c_index/val", scalar_value=c_index, global_step=epoch)

    logger.info('------------- COX LOSS ----------------')
    logger.info(f'Loss mean: {losses.avg}')
    logger.info('---------------  scores ---------------')
    logger.info(f'Concordance-index: {c_index}')
    logger.info(f'best_perf: {best_perf}')
    logger.info('--------------- ------- ---------------')
    perf = c_index
    return perf


def test(model, valid_dataset, logger, config, best_perf):
    model.eval()

    loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )
    risks, ostimes, deaths = [], [], []
    for idx, data_dict in enumerate(loader):
        data = data_dict['data']
        data_def = data_dict['data_def']
        death = data_dict['death']
        ostime = data_dict['ostime']
        info = data_dict['info']
        masks = torch.zeros(4, dtype=torch.int)
        if (data[:, 4:8].sum(dim=1) == 0).all():
            masks[1] = 1
        if (data[:, 8:12].sum(dim=1) == 0).all():
            masks[2] = 1
        if (data[:, 12:16].sum(dim=1) == 0).all():
            masks[3] = 1
        masks = masks.repeat(len(config.TRAIN.DEVICES), 1)

        data = data.cuda()
        data_def = data_def.cuda()
        death = death.cuda()
        ostime = ostime.cuda()
        info = info.cuda()
        masks = masks.cuda()
        # run training
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float32):
                risk, _, _ = model(data, data_def, info, masks)
                risks.append(risk[:, -1])
                ostimes.append(ostime)
                deaths.append(death)
    if risks[-1].shape:
        risks = torch.cat(risks)
        ostimes = torch.cat(ostimes)
        deaths = torch.cat(deaths)
    else:
        risks = torch.cat(risks[:-1])
        ostimes = torch.cat(ostimes[:-1])
        deaths = torch.cat(deaths[:-1])

    ostimes, indices = torch.sort(ostimes)
    risks = risks[indices]
    deaths = deaths[indices]

    c_index = concordance_index(risks, ostimes, deaths).item()
    if c_index > best_perf:
        best_perf = c_index

    logger.info('------------- test scores -------------')
    logger.info(f'Concordance-index: {c_index}')
    logger.info(f'best_perf: {best_perf}')
    logger.info('--------------- ------- ---------------')
    perf = c_index
    return perf
