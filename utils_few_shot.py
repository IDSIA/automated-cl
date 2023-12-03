import torch


def eval_model(model, eval_dataloader, num_steps, num_classes, device='cuda'):
    running_correct = 0
    one_ = 0
    two_ = 0
    five_ = 0
    eight_ = 0
    ten_ = 0

    running_total = 0
    one_running_total = 0

    for val_batch_id, val_batch in enumerate(eval_dataloader):
        val_inputs, val_targets = val_batch['train']
        val_inputs = val_inputs.to(device=device)  # (B, len, 1, 28, 28)
        val_targets = val_targets.to(device=device)  # (B, len)
        val_bsz, val_len = val_targets.shape

        val_rand_mapping = [torch.randperm(val_len) for i in range(val_bsz)]

        # shuffle meta-test batch
        shuffled_val_inputs = []
        shuffled_val_targets = []

        batch_id = 0
        for map_for_batch in val_rand_mapping:
            shuffled_val_inputs.append(val_inputs[batch_id][map_for_batch])
            shuffled_val_targets.append(val_targets[batch_id][map_for_batch])
            batch_id += 1

        val_inputs = torch.stack(shuffled_val_inputs, dim=1)
        val_targets = torch.stack(shuffled_val_targets, dim=1)

        with torch.no_grad():
            delayed_labels = val_targets[:-1]
            dummy_first_token = torch.zeros_like(delayed_labels[0].unsqueeze(0))
            label_feedback = torch.cat([dummy_first_token, delayed_labels], dim=0)
            outputs, _ = model(val_inputs, label_feedback)
            _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == val_targets)

            running_correct += bool_correct_pred.sum().item()
            one_running_total += val_bsz * num_classes
            running_total += val_bsz * val_len

            val_targets = val_targets.transpose(0, 1)
            bool_correct_pred = bool_correct_pred.transpose(0, 1)

            for b in range(val_bsz):
                prev_cl_end = 0
                _, cnts_uniq = torch.unique(
                    val_targets[b], sorted=True, return_counts=True)
                _, indices = torch.sort(val_targets[b], stable=True)
                for cl in range(num_classes):
                    cl_cnts = cnts_uniq[cl]
                    cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                    cl_indices = cl_indices[:10]  # compute up to 10 occ.
                    prev_cl_end = prev_cl_end + cl_cnts

                    one_ += bool_correct_pred[b][cl_indices[0]]
                    two_ += bool_correct_pred[b][cl_indices[1]]
                    five_ += bool_correct_pred[b][cl_indices[4]]
                    eight_ += bool_correct_pred[b][cl_indices[7]]
                    ten_ += bool_correct_pred[b][cl_indices[9]]

        if val_batch_id > num_steps:
            break

    one_ = one_ / one_running_total
    two_ = two_ / one_running_total
    five_ = five_ / one_running_total
    eight_ = eight_ / one_running_total
    ten_ = ten_ / one_running_total

    running_correct = running_correct / running_total

    return running_correct, one_, two_, five_, eight_, ten_


# one shot
# sync input, only last position for evaluation
def eval_model_one_shot(model, eval_dataloader, num_steps, device='cuda'):
    val_running_correct = 0
    val_running_total = 0

    for val_batch_id, val_batch in enumerate(eval_dataloader):
        val_inputs, val_targets = val_batch['train']
        val_inputs = val_inputs.to(device=device)  # (B, len, 1, 28, 28)
        val_targets = val_targets.to(device=device)  # (B, len)
        val_bsz, val_len = val_targets.shape

        val_rand_mapping = [torch.randperm(val_len) for i in range(val_bsz)]

        # shuffle meta-test batch
        shuffled_val_inputs = []
        shuffled_val_targets = []

        batch_id = 0
        for map_for_batch in val_rand_mapping:
            shuffled_val_inputs.append(val_inputs[batch_id][map_for_batch])
            shuffled_val_targets.append(val_targets[batch_id][map_for_batch])
            batch_id += 1

        val_inputs = torch.stack(shuffled_val_inputs, dim=1)
        val_targets = torch.stack(shuffled_val_targets, dim=1)

        # 'test' part
        val_test_inputs, val_test_targets = val_batch['test']
        val_test_inputs = val_test_inputs.to(device=device)  # (B, len, 1, 28, 28)
        val_test_targets = val_test_targets.to(device=device)  # (B, len)
        val_bsz, val_len = val_test_targets.shape

        val_test_rand_mapping = [torch.randperm(val_len) for i in range(val_bsz)]

        # shuffle meta-test batch
        shuffled_val_test_inputs = []
        shuffled_val_test_targets = []

        batch_id = 0
        for map_for_batch in val_test_rand_mapping:
            shuffled_val_test_inputs.append(val_test_inputs[batch_id][map_for_batch])
            shuffled_val_test_targets.append(val_test_targets[batch_id][map_for_batch])
            batch_id += 1

        val_test_inputs = torch.stack(shuffled_val_test_inputs, dim=1)
        val_test_targets = torch.stack(shuffled_val_test_targets, dim=1)

        val_test_inputs = val_test_inputs[0].unsqueeze(0)
        val_test_targets = val_test_targets[0].unsqueeze(0)

        val_net_input = torch.cat([val_inputs, val_test_inputs], dim=0)
        val_target_labels = torch.cat([val_targets, val_test_targets], dim=0)

        with torch.no_grad():
            sync_labels = val_target_labels[:-1]
            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)
            outputs, _ = model(val_net_input, label_feedback)
            outputs = outputs[-1]
            _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == val_target_labels[-1])

            val_running_correct += bool_correct_pred.sum().item()
            val_running_total += val_bsz

        if val_batch_id > num_steps:
            break

    running_correct = val_running_correct / val_running_total

    return running_correct


def eval_pretrain_model(model, eval_dataloader, num_steps, device='cuda'):
    val_running_correct = 0
    val_running_total = 0

    for val_batch_id, val_batch in enumerate(eval_dataloader):
        val_inputs, val_targets = val_batch['train']
        val_inputs = val_inputs.to(device=device)  # (B, len, **)
        val_targets = val_targets.to(device=device)  # (B, len)

        val_shape = val_inputs.shape
        bsz, slen = val_shape[0], val_shape[1]
        val_inputs = val_inputs.view(bsz * slen, *val_shape[2:])
        val_targets = val_targets.view(bsz * slen)

        # 'test' part
        val_test_inputs, val_test_targets = val_batch['test']
        val_test_inputs = val_test_inputs.to(device=device)  # (B, len, **)
        val_test_targets = val_test_targets.to(device=device)  # (B, len)

        val_test_shape = val_test_inputs.shape
        tlen = val_test_shape[1]
        val_test_inputs = val_test_inputs.view(bsz * tlen, *val_test_shape[2:])
        val_test_targets = val_test_targets.view(bsz * tlen)

        val_net_input = torch.cat([val_inputs, val_test_inputs], dim=0)
        val_target_labels = torch.cat([val_targets, val_test_targets], dim=0)

        with torch.no_grad():
            outputs = model(val_net_input)
            _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == val_target_labels)

            val_running_correct += bool_correct_pred.sum().item()
            val_running_total += bsz * (slen + tlen)

        if val_batch_id > num_steps:
            break

    running_correct = val_running_correct / val_running_total

    return running_correct

plot_index = 0

def plot_images(tensor):
    global plot_index
    from matplotlib import pyplot as plt
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    slen, c, d1, d2 = tensor.shape
    num_cols = slen
    num_rows = 1
    tensor = tensor.permute(0, 2, 3, 1).to('cpu').numpy()
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        # ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(f'input_images_{plot_index}.png')


# set save_images to try
def eval_acl_ab_model_label_sync(model, eval_dataloader_a, eval_dataloader_b,
                                 num_steps, device='cuda', state=None):
    val_running_correct1 = 0
    val_running_total1 = 0

    val_running_correct2 = 0
    val_running_total2 = 0

    val_running_correct_acl = 0
    val_running_total_acl = 0

    for val_batch_id, (batch_1, batch_2) in enumerate(zip(eval_dataloader_a, eval_dataloader_b)):
        # TASK A
        val_inputs1, val_targets1 = batch_1['train']
        val_inputs1 = val_inputs1.to(device=device)  # (B, len, **)
        val_targets1 = val_targets1.to(device=device)  # (B, len)
        val_bsz, _ = val_targets1.shape

        val_inputs1 = val_inputs1.transpose(0, 1)
        val_targets1 = val_targets1.transpose(0, 1)

        # 'test' part
        val_test_inputs1, val_test_targets1 = batch_1['test']
        val_test_inputs1 = val_test_inputs1.to(device=device)  # (B, len, **)
        val_test_targets1 = val_test_targets1.to(device=device)  # (B, len)

        val_test_inputs1 = val_test_inputs1.transpose(0, 1)
        val_test_targets1 = val_test_targets1.transpose(0, 1)

        # take just one element
        val_test_inputs1 = val_test_inputs1[0].unsqueeze(0)
        val_test_targets1 = val_test_targets1[0].unsqueeze(0)

        # TASK B
        val_inputs2, val_targets2 = batch_2['train']
        val_inputs2 = val_inputs2.to(device=device)  # (B, len, **)
        val_targets2 = val_targets2.to(device=device)  # (B, len)
        val_bsz2, _ = val_targets2.shape

        if val_bsz != val_bsz2:
            break

        val_inputs2 = val_inputs2.transpose(0, 1)
        val_targets2 = val_targets2.transpose(0, 1)

        # 'test' part
        val_test_inputs2, val_test_targets2 = batch_2['test']
        val_test_inputs2 = val_test_inputs2.to(device=device)  # (B, len, **)
        val_test_targets2 = val_test_targets2.to(device=device)  # (B, len)

        val_test_inputs2 = val_test_inputs2.transpose(0, 1)
        val_test_targets2 = val_test_targets2.transpose(0, 1)

        # take just one element
        val_test_inputs2 = val_test_inputs2[0].unsqueeze(0)
        val_test_targets2 = val_test_targets2[0].unsqueeze(0)

        # forward TASK A
        # val_net_input = torch.cat([val_inputs1, val_test_inputs1], dim=0)
        val_target_labels = torch.cat([val_targets1, val_test_targets1], dim=0)

        with torch.no_grad():
            sync_labels = val_target_labels[:-1]
            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes
            # label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)\
            # get state first
            if state is not None:
                _, out_state = model(val_inputs1, sync_labels, state)
            else:
                _, out_state = model(val_inputs1, sync_labels)

            # actual eval
            outputs, _ = model(val_test_inputs1, dummy_last_token, out_state)
            out_state = model.clone_state(out_state)

            outputs = outputs[-1]
            _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == val_test_targets1[-1])

            val_running_correct1 += bool_correct_pred.sum().item()
            val_running_total1 += val_bsz

        # forward TASK B
        # val_net_input = torch.cat([val_inputs2, val_test_inputs2], dim=0)
        val_target_labels = torch.cat([val_targets2, val_test_targets2], dim=0)

        with torch.no_grad():
            sync_labels = val_target_labels[:-1]
            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes
            # label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)\
            # get state first
            _, out_state = model(val_inputs2, val_targets2, out_state)
            out_state = model.clone_state(out_state)

            # actual eval
            outputs, _ = model(val_test_inputs2, dummy_last_token, out_state)
            
            outputs = outputs[-1]
            _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == val_test_targets2[-1])

            val_running_correct2 += bool_correct_pred.sum().item()

            val_running_total2 += val_bsz

        # forward ACL
        with torch.no_grad():
            outputs, _ = model(val_test_inputs1, dummy_last_token, out_state)
            
            outputs = outputs[-1]
            _, predicted = outputs.max(-1)
            bool_correct_pred_acl = (predicted == val_test_targets1[-1])

            val_running_correct_acl += bool_correct_pred_acl.sum().item()
            val_running_total_acl += val_bsz

        if val_batch_id > num_steps:
            break

    running_correct1 = val_running_correct1 / val_running_total1
    running_correct2 = val_running_correct2 / val_running_total2
    running_correct_acl = val_running_correct_acl / val_running_total_acl

    return (running_correct1, running_correct2, running_correct_acl)


# set save_images to try
def eval_model_label_sync(model, eval_dataloader, num_steps, device='cuda',
                          save_images=False, state=None, get_state=False,
                          get_second_last_state=False, eval_no_context=False):

    if eval_no_context:
        eval_model_label_sync_no_context(
            model, eval_dataloader, num_steps, device, state, get_state, get_second_last_state)
    val_running_correct = 0
    val_running_total = 0

    for val_batch_id, val_batch in enumerate(eval_dataloader):
        val_inputs, val_targets = val_batch['train']
        val_inputs = val_inputs.to(device=device)  # (B, len, **)
        val_targets = val_targets.to(device=device)  # (B, len)
        val_bsz, _ = val_targets.shape

        val_inputs = val_inputs.transpose(0, 1)
        val_targets = val_targets.transpose(0, 1)

        # 'test' part
        val_test_inputs, val_test_targets = val_batch['test']
        val_test_inputs = val_test_inputs.to(device=device)  # (B, len, **)
        val_test_targets = val_test_targets.to(device=device)  # (B, len)

        val_test_inputs = val_test_inputs.transpose(0, 1)
        val_test_targets = val_test_targets.transpose(0, 1)

        # take just one element
        val_test_inputs = val_test_inputs[0].unsqueeze(0)
        val_test_targets = val_test_targets[0].unsqueeze(0)

        val_net_input = torch.cat([val_inputs, val_test_inputs], dim=0)
        if save_images:
            slen, bsz, c, d1, d2 = val_net_input.shape
            plot_images(val_net_input.reshape(slen * bsz, c, d1, d2))
        val_target_labels = torch.cat([val_targets, val_test_targets], dim=0)

        with torch.no_grad():
            sync_labels = val_target_labels[:-1]
            if save_images:
                print(sync_labels)
            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes
            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)
            if state is not None:
                outputs, out_state = model(val_net_input, label_feedback, state)
            else:
                outputs, out_state = model(val_net_input, label_feedback)
            outputs = outputs[-1]
            _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == val_target_labels[-1])

            val_running_correct += bool_correct_pred.sum().item()
            val_running_total += val_bsz

        if val_batch_id > num_steps:
            break

    running_correct = val_running_correct / val_running_total

    if get_second_last_state:
        with torch.no_grad():
            _, out_state = model(val_inputs, val_targets, state)
        return running_correct, out_state
    elif get_state:
        return running_correct, out_state
    else:
        return running_correct


# set save_images to try
def eval_model_label_sync_no_context(
        model, eval_dataloader, num_steps, device='cuda', state=None,
        get_state=False, get_second_last_state=False):

    val_running_correct = 0
    val_running_total = 0

    for val_batch_id, val_batch in enumerate(eval_dataloader):
        # 'test' part
        val_test_inputs, val_target_labels = val_batch['test']
        val_test_inputs = val_test_inputs.to(device=device)  # (B, len, **)
        val_target_labels = val_target_labels.to(device=device)  # (B, len)
        val_bsz, _ = val_target_labels.shape

        val_test_inputs = val_test_inputs.transpose(0, 1)
        val_target_labels = val_target_labels.transpose(0, 1)

        # take just one element
        val_test_inputs = val_test_inputs[0].unsqueeze(0)
        val_target_labels = val_target_labels[0].unsqueeze(0)

        with torch.no_grad():
            label_feedback = torch.zeros_like(val_target_labels[0].unsqueeze(0))
            if model.extra_label:
                label_feedback = label_feedback + model.num_classes
            if state is not None:
                outputs, out_state = model(val_test_inputs, label_feedback, state)
            else:
                outputs, out_state = model(val_test_inputs, label_feedback)
            outputs = outputs[-1]
            _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == val_target_labels[-1])

            val_running_correct += bool_correct_pred.sum().item()
            val_running_total += val_bsz

        if val_batch_id > num_steps:
            break

    running_correct = val_running_correct / val_running_total

    if get_second_last_state:
        with torch.no_grad():
            _, out_state = model(val_inputs, val_targets, state)
        return running_correct, out_state
    elif get_state:
        return running_correct, out_state
    else:
        return running_correct


def eval_model_delayed_label(model, eval_dataloader, num_steps, n_way,
                             device='cuda', state=None):
    val_running_correct = 0
    val_running_total = 0

    acc_per_shot = {'0': 0, '1': 0, '5': 0, '8': 0, '10': 0}

    one_running_total = 0

    for val_batch_id, val_batch in enumerate(eval_dataloader):
        val_inputs, val_targets = val_batch['train']
        val_inputs = val_inputs.to(device=device)  # (B, len, **)
        val_targets = val_targets.to(device=device)  # (B, len)
        val_bsz, val_len = val_targets.shape

        val_inputs = val_inputs.transpose(0, 1)
        val_targets = val_targets.transpose(0, 1)

        with torch.no_grad():
            delayed_labels = val_targets[:-1]
            dummy_last_token = torch.zeros_like(delayed_labels[0].unsqueeze(0))
            label_feedback = torch.cat(
                [dummy_last_token, delayed_labels], dim=0)
            outputs, _ = model(val_inputs, label_feedback, state)
            _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == val_targets)

            val_running_correct += bool_correct_pred.sum().item()
            val_running_total += val_bsz * val_len
            one_running_total += val_bsz * n_way  # for each class

            val_targets = val_targets.transpose(0, 1)
            bool_correct_pred = bool_correct_pred.transpose(0, 1)

            # compute shot-wise accuracy
            # NB: this is averaged over classes. Not comparable to k-shot
            # accuracy since e.g. the last class which appears for the first 
            # time should have a higher accuracy.
            for b in range(val_bsz):
                prev_cl_end = 0
                _, cnts_uniq = torch.unique(
                    val_targets[b], sorted=True, return_counts=True)
                _, indices = torch.sort(val_targets[b], stable=True)
                for cl in range(n_way):
                    cl_cnts = cnts_uniq[cl]
                    cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                    cl_indices = cl_indices[:11]  # compute up to 10 occ. TODO assert k is larger than 10
                    prev_cl_end = prev_cl_end + cl_cnts

                    acc_per_shot['0'] += bool_correct_pred[b][cl_indices[0]]
                    acc_per_shot['1'] += bool_correct_pred[b][cl_indices[1]]
                    acc_per_shot['5'] += bool_correct_pred[b][cl_indices[5]]
                    acc_per_shot['8'] += bool_correct_pred[b][cl_indices[8]]
                    acc_per_shot['10'] += bool_correct_pred[b][cl_indices[10]]

        if val_batch_id > num_steps:
            break

    running_correct = val_running_correct / val_running_total

    acc_per_shot['0'] = acc_per_shot['0'] / one_running_total
    acc_per_shot['1'] = acc_per_shot['1'] / one_running_total
    acc_per_shot['5'] = acc_per_shot['5'] / one_running_total
    acc_per_shot['8'] = acc_per_shot['8'] / one_running_total
    acc_per_shot['10'] = acc_per_shot['10'] / one_running_total

    return running_correct, acc_per_shot


def eval_model_delayed_label_v2(model, eval_dataloader, num_steps, n_way, 
                                k_shot, device='cuda', state=None):
    val_running_correct = 0
    val_running_total = 0

    acc_per_shot = []
    cnt_per_shot = []

    for _ in range(k_shot):
        acc_per_shot.append(0)
        cnt_per_shot.append(0)

    for val_batch_id, val_batch in enumerate(eval_dataloader):
        val_inputs, val_targets = val_batch['train']
        val_inputs = val_inputs.to(device=device)  # (B, len, **)
        val_targets = val_targets.to(device=device)  # (B, len)
        val_bsz, val_len = val_targets.shape

        val_inputs = val_inputs.transpose(0, 1)
        val_targets = val_targets.transpose(0, 1)

        with torch.no_grad():
            delayed_labels = val_targets[:-1]
            dummy_last_token = torch.zeros_like(delayed_labels[0].unsqueeze(0))
            label_feedback = torch.cat(
                [dummy_last_token, delayed_labels], dim=0)
            outputs, _ = model(val_inputs, label_feedback, state)
            _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == val_targets)

            val_running_correct += bool_correct_pred.sum().item()
            val_running_total += val_bsz * val_len

            val_targets = val_targets.transpose(0, 1)
            bool_correct_pred = bool_correct_pred.transpose(0, 1)

            # compute shot-wise accuracy
            # NB: this is averaged over classes. Not comparable to k-shot
            # accuracy since e.g. the last class which appears for the first 
            # time should have a higher accuracy.
            for b in range(val_bsz):
                prev_cl_end = 0
                _, cnts_uniq = torch.unique(
                    val_targets[b], sorted=True, return_counts=True)
                _, indices = torch.sort(val_targets[b], stable=True)
                for cl in range(n_way):
                    cl_cnts = cnts_uniq[cl]
                    cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                    cl_indices_len = len(cl_indices)
                    prev_cl_end = prev_cl_end + cl_cnts

                    for shot in range(k_shot):
                        if cl_indices_len > shot:
                            acc_per_shot[shot] += (
                                bool_correct_pred[b][cl_indices[shot]].item())
                            cnt_per_shot[shot] += 1

        if val_batch_id > num_steps:
            break

    running_correct = 100 * val_running_correct / val_running_total

    for shot in range(k_shot):
        shot_acc = (
            100 * acc_per_shot[shot] / cnt_per_shot[shot]
        )
        acc_per_shot[shot] = shot_acc

    return running_correct, acc_per_shot


# eval without sequence
def eval_model_ctrl(model, eval_dataloader, num_steps, loginf=None,
                                 device='cuda', state=None, split='train'):
    val_running_correct = 0
    val_running_total = 0
    assert loginf is not None

    # acc_per_shot = []
    # cnt_per_shot = []

    # for _ in range(k_shot):
    #     acc_per_shot.append(0)
    #     cnt_per_shot.append(0)

    # batch size is essentially 1. 
    # the original batch size becomes the sequence length
    for val_batch_id, cur_batch in enumerate(eval_dataloader):
        val_inputs, val_targets = cur_batch

        val_inputs = val_inputs.to(device=device)  # (B, **)
        val_targets = val_targets.to(device=device)  # (B,)

        # from torchvision.utils import save_image

        # for k in range(0, 20):
        # # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
        #     save_image(val_inputs[k], f'img{k}.png')
        #     print(val_targets[k])
        # import sys; sys.exit()

        val_inputs = val_inputs.unsqueeze(0)
        val_targets = val_targets.unsqueeze(0)  # add artificial batch dim
        val_bsz, val_len = val_targets.shape

        val_inputs = val_inputs.transpose(0, 1)
        val_targets = val_targets.transpose(0, 1)

        with torch.no_grad():
            delayed_labels = val_targets[:-1]
            dummy_last_token = torch.zeros_like(delayed_labels[0].unsqueeze(0))
            label_feedback = torch.cat(
                [dummy_last_token, delayed_labels], dim=0)
            outputs, _ = model(val_inputs, label_feedback, state)

            _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == val_targets)

            val_running_correct += bool_correct_pred.sum().item()
            val_running_total += val_bsz * val_len

            loginf(f'batch {val_batch_id}, {split} accuracy: {100 * val_running_correct / val_running_total}')

            # val_targets = val_targets.transpose(0, 1)
            # bool_correct_pred = bool_correct_pred.transpose(0, 1)

            # compute shot-wise accuracy
            # NB: this is averaged over classes. Not comparable to k-shot
            # accuracy since e.g. the last class which appears for the first 
            # time should have a higher accuracy.
            # for b in range(val_bsz):
            #     prev_cl_end = 0
            #     _, cnts_uniq = torch.unique(
            #         val_targets[b], sorted=True, return_counts=True)
            #     _, indices = torch.sort(val_targets[b], stable=True)
            #     for cl in range(n_way):
            #         cl_cnts = cnts_uniq[cl]
            #         cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
            #         cl_indices_len = len(cl_indices)
            #         prev_cl_end = prev_cl_end + cl_cnts

                    # for shot in range(k_shot):
                    #     if cl_indices_len > shot:
                    #         acc_per_shot[shot] += (
                    #             bool_correct_pred[b][cl_indices[shot]].item())
                    #         cnt_per_shot[shot] += 1

        if val_batch_id > num_steps:
            break

    running_correct = 100 * val_running_correct / val_running_total

    # for shot in range(k_shot):
    #     shot_acc = (
    #         100 * acc_per_shot[shot] / cnt_per_shot[shot]
    #     )
    #     acc_per_shot[shot] = shot_acc

    return running_correct


def eval_model_delayed_label_ctrl(model, eval_dataloader, num_steps, loginf=None,
                                  device='cuda', state=None, split='train'):
    val_running_correct = 0
    val_running_total = 0
    assert loginf is not None

    # acc_per_shot = []
    # cnt_per_shot = []

    # for _ in range(k_shot):
    #     acc_per_shot.append(0)
    #     cnt_per_shot.append(0)

    # batch size is essentially 1. 
    # the original batch size becomes the sequence length
    for val_batch_id, cur_batch in enumerate(eval_dataloader):
        val_inputs, val_targets = cur_batch

        val_inputs = val_inputs.to(device=device)  # (B, **)
        val_targets = val_targets.to(device=device)  # (B,)

        # from torchvision.utils import save_image

        # for k in range(0, 20):
        # # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
        #     save_image(val_inputs[k], f'img{k}.png')
        #     print(val_targets[k])
        # import sys; sys.exit()

        val_inputs = val_inputs.unsqueeze(0)
        val_targets = val_targets.unsqueeze(0)  # add artificial batch dim
        val_bsz, val_len = val_targets.shape

        val_inputs = val_inputs.transpose(0, 1)
        val_targets = val_targets.transpose(0, 1)

        # TODO debugging
        state = None

        with torch.no_grad():
            delayed_labels = val_targets[:-1]
            dummy_last_token = torch.zeros_like(delayed_labels[0].unsqueeze(0))
            label_feedback = torch.cat(
                [dummy_last_token, delayed_labels], dim=0)
            outputs, state = model(val_inputs, label_feedback, state)

            _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == val_targets)

            val_running_correct += bool_correct_pred.sum().item()
            val_running_total += val_bsz * val_len

            loginf(f'batch {val_batch_id}, {split} accuracy: {100 * val_running_correct / val_running_total}')

            # val_targets = val_targets.transpose(0, 1)
            # bool_correct_pred = bool_correct_pred.transpose(0, 1)

            # compute shot-wise accuracy
            # NB: this is averaged over classes. Not comparable to k-shot
            # accuracy since e.g. the last class which appears for the first 
            # time should have a higher accuracy.
            # for b in range(val_bsz):
            #     prev_cl_end = 0
            #     _, cnts_uniq = torch.unique(
            #         val_targets[b], sorted=True, return_counts=True)
            #     _, indices = torch.sort(val_targets[b], stable=True)
            #     for cl in range(n_way):
            #         cl_cnts = cnts_uniq[cl]
            #         cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
            #         cl_indices_len = len(cl_indices)
            #         prev_cl_end = prev_cl_end + cl_cnts

                    # for shot in range(k_shot):
                    #     if cl_indices_len > shot:
                    #         acc_per_shot[shot] += (
                    #             bool_correct_pred[b][cl_indices[shot]].item())
                    #         cnt_per_shot[shot] += 1

        if val_batch_id > num_steps:
            break

    running_correct = 100 * val_running_correct / val_running_total

    # for shot in range(k_shot):
    #     shot_acc = (
    #         100 * acc_per_shot[shot] / cnt_per_shot[shot]
    #     )
    #     acc_per_shot[shot] = shot_acc

    return running_correct, state



# hard coded for two tasks
def eval_model_delayed_label_multi_sequential(
        model, eval_dataloader0, eval_dataloader1, num_steps, n_way, k_shot,
        device='cuda', state=None):

    running_correct = 0
    running_total = 0

    task_running_correct = {0: 0., 1: 0.}
    counts = 0

    acc_per_shot = {0: [], 1: []}
    cnt_per_shot = {0: [], 1: []}

    for key in acc_per_shot.keys():
        for _ in range(k_shot):
            acc_per_shot[key].append(0)
            cnt_per_shot[key].append(0)

    for batch_id, (batch0, batch1) in enumerate(zip(eval_dataloader0, eval_dataloader1)):
        val_inputs0, val_targets0 = batch0['train']
        val_inputs1, val_targets1 = batch1['train']
        del batch0['test'], batch1['test']

        val_inputs0 = val_inputs0.to(device=device)  # (B, len, **)
        val_targets0 = val_targets0.to(device=device)  # (B, len)
        val_bsz0, val_len0 = val_targets0.shape

        val_inputs1 = val_inputs1.to(device=device)  # (B, len, **)
        val_targets1 = val_targets1.to(device=device)  # (B, len)
        val_bsz1, val_len1 = val_targets1.shape

        val_inputs0 = val_inputs0.transpose(0, 1)
        val_targets0 = val_targets0.transpose(0, 1)

        val_inputs1 = val_inputs1.transpose(0, 1)
        val_targets1 = val_targets1.transpose(0, 1)

        # no trimming needed for eval.
        # contenate along time dimension, alternate order
        if batch_id % 2 == 0:  # ID 0 first
            net_input = torch.cat([val_inputs0, val_inputs1], dim=0)
            target_labels = torch.cat([val_targets0, val_targets1], dim=0)
        else:  # miniimagenet first
            net_input = torch.cat([val_inputs1, val_inputs0], dim=0)
            target_labels = torch.cat([val_targets1, val_targets0], dim=0)

        slen, bsz = target_labels.shape

        delayed_labels = target_labels[:-1]
        dummy_last_token = torch.zeros_like(delayed_labels[0].unsqueeze(0))
        label_feedback = torch.cat([dummy_last_token, delayed_labels], dim=0)

        outputs, _ = model(net_input, label_feedback, state)
        _, predicted = outputs.max(-1)
        bool_correct_pred = (predicted == target_labels)

        running_correct += bool_correct_pred.sum().item()
        running_total += slen * bsz

        if batch_id % 2 == 0:  # ID 0 first
            bool_correct_pred0 = bool_correct_pred[:val_len0]
            bool_correct_pred1 = bool_correct_pred[val_len0:]
        else:
            bool_correct_pred1 = bool_correct_pred[:val_len1]
            bool_correct_pred0 = bool_correct_pred[val_len1:]

        task_running_correct[0] += bool_correct_pred0.sum().item()
        task_running_correct[1] += bool_correct_pred1.sum().item()

        assert val_bsz0 == val_bsz1
        assert val_len0 == val_len1
        counts += val_bsz0 * val_len0  # same size

        val_targets0 = val_targets0.transpose(0, 1)
        val_targets1 = val_targets1.transpose(0, 1)

        bool_correct_pred0 = bool_correct_pred0.transpose(0, 1)
        bool_correct_pred1 = bool_correct_pred1.transpose(0, 1)

        for b in range(bsz):
            # task 0
            prev_cl_end = 0
            _, cnts_uniq = torch.unique(
                val_targets0[b], sorted=True, return_counts=True)
            _, indices = torch.sort(val_targets0[b], stable=True)
            for cl in range(n_way):
                cl_cnts = cnts_uniq[cl]
                cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                cl_indices_len = len(cl_indices)
                prev_cl_end += cl_cnts

                for shot in range(k_shot):
                    if cl_indices_len > shot:
                        acc_per_shot[0][shot] += (
                            bool_correct_pred0[b][cl_indices[shot]].item())
                        cnt_per_shot[0][shot] += 1
            # task 1
            prev_cl_end = 0
            _, cnts_uniq = torch.unique(
                val_targets1[b], sorted=True, return_counts=True)
            _, indices = torch.sort(val_targets1[b], stable=True)
            for cl in range(n_way):
                cl_cnts = cnts_uniq[cl]
                cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                cl_indices_len = len(cl_indices)
                prev_cl_end += cl_cnts

                for shot in range(k_shot):
                    if cl_indices_len > shot:
                        acc_per_shot[1][shot] += (
                            bool_correct_pred1[b][cl_indices[shot]].item())
                        cnt_per_shot[1][shot] += 1

        if batch_id > num_steps:
            break

    running_correct = 100 * running_correct / running_total
    task_running_correct[0] = 100 * task_running_correct[0] / counts
    task_running_correct[1] = 100 * task_running_correct[1] / counts

    for key in acc_per_shot.keys():
        for shot in range(k_shot):
            shot_acc = (
                100 * acc_per_shot[key][shot] / cnt_per_shot[key][shot]
            )
            acc_per_shot[key][shot] = shot_acc

    return running_correct, task_running_correct, acc_per_shot


# hard coded for two tasks, three segment
def eval_model_delayed_label_three_segments(
        model, eval_dataloader0, eval_dataloader1, num_steps, n_way, k_shot,
        half_seq_len, device='cuda', state=None):

    running_correct = 0
    running_total = 0

    task_running_correct = {0: 0., 1: 0., 2: 0., 3: 0., 10: 0., 11: 0., 12: 0., 13: 0.}
    counts = 0

    acc_per_shot = {0: [], 1: [], 2: [], 3: [], 10: [], 11: [], 12: [], 13: []}
    cnt_per_shot = {0: [], 1: [], 2: [], 3: [], 10: [], 11: [], 12: [], 13: []}

    for key in acc_per_shot.keys():
        for _ in range(k_shot):
            acc_per_shot[key].append(0)
            cnt_per_shot[key].append(0)

    for batch_id, (batch0, batch1) in enumerate(zip(eval_dataloader0, eval_dataloader1)):
        val_inputs0, val_targets0 = batch0['train']
        val_inputs1, val_targets1 = batch1['train']
        del batch0['test'], batch1['test']

        val_inputs0 = val_inputs0.to(device=device)  # (B, len, **)
        val_targets0 = val_targets0.to(device=device)  # (B, len)
        val_bsz0, val_len0 = val_targets0.shape

        val_inputs1 = val_inputs1.to(device=device)  # (B, len, **)
        val_targets1 = val_targets1.to(device=device)  # (B, len)
        val_bsz1, val_len1 = val_targets1.shape

        val_inputs0 = val_inputs0.transpose(0, 1)
        val_targets0 = val_targets0.transpose(0, 1)

        val_inputs1 = val_inputs1.transpose(0, 1)
        val_targets1 = val_targets1.transpose(0, 1)

        val_inputs0_0 = val_inputs0[:half_seq_len]
        val_inputs0_1 = val_inputs0[half_seq_len:]

        val_targets0_0 = val_targets0[:half_seq_len]
        val_targets0_1 = val_targets0[half_seq_len:]

        val_inputs1_0 = val_inputs1[:half_seq_len]
        val_inputs1_1 = val_inputs1[half_seq_len:]

        val_targets1_0 = val_targets1[:half_seq_len]
        val_targets1_1 = val_targets1[half_seq_len:]

        # no trimming needed for eval.
        # contenate along time dimension, alternate order
        order_ = batch_id % 2
        if order_ == 0:  # ID 0 first
            net_input = torch.cat([val_inputs0_0, val_inputs1_0, val_inputs0_1, val_inputs1_1], dim=0)
            target_labels = torch.cat([val_targets0_0, val_targets1_0, val_targets0_1, val_targets1_1], dim=0)
        else:  # miniimagenet first
            net_input = torch.cat([val_inputs1_0, val_inputs0_0, val_inputs1_1, val_inputs0_1], dim=0)
            target_labels = torch.cat([val_targets1_0, val_targets0_0, val_targets1_1, val_targets0_1], dim=0)

        slen, bsz = target_labels.shape

        delayed_labels = target_labels[:-1]
        dummy_last_token = torch.zeros_like(delayed_labels[0].unsqueeze(0))
        label_feedback = torch.cat([dummy_last_token, delayed_labels], dim=0)

        outputs, _ = model(net_input, label_feedback, state)
        _, predicted = outputs.max(-1)
        bool_correct_pred = (predicted == target_labels)

        running_correct += bool_correct_pred.sum().item()
        running_total += slen * bsz

        assert val_len0 % 2 == 0
        seg_len = val_len0 // 2
        bool_correct_pred0 = bool_correct_pred[:seg_len]
        bool_correct_pred1 = bool_correct_pred[seg_len:2 * seg_len]
        bool_correct_pred2 = bool_correct_pred[2 * seg_len: 3 * seg_len]
        bool_correct_pred3 = bool_correct_pred[3 * seg_len:]

        if order_ == 0:
            task_running_correct[0] += bool_correct_pred0.sum().item()
            task_running_correct[11] += bool_correct_pred1.sum().item()
            task_running_correct[2] += bool_correct_pred2.sum().item()
            task_running_correct[13] += bool_correct_pred3.sum().item()

            part0 = val_targets0_0.transpose(0, 1)
            part1 = val_targets1_0.transpose(0, 1)
            part2 = val_targets0_1.transpose(0, 1)
            part3 = val_targets1_1.transpose(0, 1)
        else:
            task_running_correct[10] += bool_correct_pred0.sum().item()
            task_running_correct[1] += bool_correct_pred1.sum().item()
            task_running_correct[12] += bool_correct_pred2.sum().item()
            task_running_correct[3] += bool_correct_pred3.sum().item()

            part0 = val_targets1_0.transpose(0, 1)
            part1 = val_targets0_0.transpose(0, 1)
            part2 = val_targets1_1.transpose(0, 1)
            part3 = val_targets0_1.transpose(0, 1)

        assert val_bsz0 == val_bsz1
        assert val_len0 == val_len1
        re_seg_len, seg_bsz = bool_correct_pred0.shape
        assert re_seg_len == seg_len
        counts += seg_bsz * seg_len // 2  # every two batches

        bool_correct_pred0 = bool_correct_pred0.transpose(0, 1)
        bool_correct_pred1 = bool_correct_pred1.transpose(0, 1)
        bool_correct_pred2 = bool_correct_pred2.transpose(0, 1)
        bool_correct_pred3 = bool_correct_pred3.transpose(0, 1)

        offset_ = 0 if order_ == 0 else 10
        # default indices are: 0, 11, 2, 13. Add or move 10
        # to get the other case: 10, 1, 12, 3.

        for b in range(bsz):
            # task 0
            prev_cl_end = 0
            _, cnts_uniq = torch.unique(
                part0[b], sorted=True, return_counts=True)
            _, indices = torch.sort(part0[b], stable=True)
            # skip the accidental case where unseen label
            if n_way <= cnts_uniq.shape[0]:
                for cl in range(n_way):
                    cl_cnts = cnts_uniq[cl]
                    cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                    cl_indices_len = len(cl_indices)
                    prev_cl_end += cl_cnts

                    for shot in range(k_shot):
                        if cl_indices_len > shot:
                            acc_per_shot[0 + offset_][shot] += (
                                bool_correct_pred0[b][cl_indices[shot]].item())
                            cnt_per_shot[0 + offset_][shot] += 1
            # task 1
            prev_cl_end = 0
            _, cnts_uniq = torch.unique(
                part1[b], sorted=True, return_counts=True)
            _, indices = torch.sort(part1[b], stable=True)
            if n_way <= cnts_uniq.shape[0]:
                for cl in range(n_way):
                    cl_cnts = cnts_uniq[cl]
                    cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                    cl_indices_len = len(cl_indices)
                    prev_cl_end += cl_cnts

                    for shot in range(k_shot):
                        if cl_indices_len > shot:
                            acc_per_shot[11 - offset_][shot] += (
                                bool_correct_pred1[b][cl_indices[shot]].item())
                            cnt_per_shot[11 - offset_][shot] += 1

            # task 2
            prev_cl_end = 0
            _, cnts_uniq = torch.unique(
                part2[b], sorted=True, return_counts=True)
            _, indices = torch.sort(part2[b], stable=True)
            if n_way <= cnts_uniq.shape[0]:
                for cl in range(n_way):
                    cl_cnts = cnts_uniq[cl]
                    cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                    cl_indices_len = len(cl_indices)
                    prev_cl_end += cl_cnts

                    for shot in range(k_shot):
                        if cl_indices_len > shot:
                            acc_per_shot[2 + offset_][shot] += (
                                bool_correct_pred2[b][cl_indices[shot]].item())
                            cnt_per_shot[2 + offset_][shot] += 1

            # task 3
            prev_cl_end = 0
            _, cnts_uniq = torch.unique(
                part3[b], sorted=True, return_counts=True)
            _, indices = torch.sort(part3[b], stable=True)
            if n_way <= cnts_uniq.shape[0]:
                for cl in range(n_way):
                    cl_cnts = cnts_uniq[cl]
                    cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                    cl_indices_len = len(cl_indices)
                    prev_cl_end += cl_cnts

                    for shot in range(k_shot):
                        if cl_indices_len > shot:
                            acc_per_shot[13 - offset_][shot] += (
                                bool_correct_pred3[b][cl_indices[shot]].item())
                            cnt_per_shot[13 - offset_][shot] += 1

        if batch_id > num_steps:
            break

    running_correct = 100 * running_correct / running_total
    for key in task_running_correct.keys():
        task_running_correct[key] = 100 * task_running_correct[key] / counts

    for key in acc_per_shot.keys():
        for shot in range(k_shot):
            shot_acc = (
                100 * acc_per_shot[key][shot] / cnt_per_shot[key][shot]
            )
            acc_per_shot[key][shot] = shot_acc

    return running_correct, task_running_correct, acc_per_shot


# hard coded for two tasks
def eval_per_pos_model_delayed_label_multi_sequential(
        model, eval_dataloader0, eval_dataloader1, num_steps, n_way, k_shot,
        device='cuda', state=None, omniglot_first=True):

    running_correct = 0
    running_total = 0

    task_running_correct = {0: 0., 1: 0.}
    counts = 0

    acc_per_shot = {0: [], 1: []}  # per positions in this case
    cnt_per_shot = {0: [], 1: []}

    for key in acc_per_shot.keys():
        for _ in range(k_shot):
            acc_per_shot[key].append(0)
            cnt_per_shot[key].append(0)

    acc_per_pos = []  # per positions in this case
    cnt_per_pos = 0

    for _ in range(k_shot * n_way * 2):
        acc_per_pos.append(0)

    for batch_id, (batch0, batch1) in enumerate(zip(eval_dataloader0, eval_dataloader1)):
        val_inputs0, val_targets0 = batch0['train']
        val_inputs1, val_targets1 = batch1['train']
        del batch0['test'], batch1['test']

        val_inputs0 = val_inputs0.to(device=device)  # (B, len, **)
        val_targets0 = val_targets0.to(device=device)  # (B, len)
        val_bsz0, val_len0 = val_targets0.shape

        val_inputs1 = val_inputs1.to(device=device)  # (B, len, **)
        val_targets1 = val_targets1.to(device=device)  # (B, len)
        val_bsz1, val_len1 = val_targets1.shape

        val_inputs0 = val_inputs0.transpose(0, 1)
        val_targets0 = val_targets0.transpose(0, 1)

        val_inputs1 = val_inputs1.transpose(0, 1)
        val_targets1 = val_targets1.transpose(0, 1)

        # no trimming needed for eval.
        # contenate along time dimension, alternate order
        if omniglot_first:  # ID 0 first
            net_input = torch.cat([val_inputs0, val_inputs1], dim=0)
            target_labels = torch.cat([val_targets0, val_targets1], dim=0)
        else:  # miniimagenet first
            net_input = torch.cat([val_inputs1, val_inputs0], dim=0)
            target_labels = torch.cat([val_targets1, val_targets0], dim=0)

        slen, bsz = target_labels.shape

        delayed_labels = target_labels[:-1]
        dummy_last_token = torch.zeros_like(delayed_labels[0].unsqueeze(0))
        label_feedback = torch.cat([dummy_last_token, delayed_labels], dim=0)

        outputs, _ = model(net_input, label_feedback, state)
        _, predicted = outputs.max(-1)
        bool_correct_pred = (predicted == target_labels)

        running_correct += bool_correct_pred.sum().item()
        running_total += slen * bsz

        # per position stats:
        assert slen == k_shot * n_way * 2
        for pos in range(k_shot * n_way * 2):
            acc_per_pos[pos] += bool_correct_pred[pos].sum().item()
        cnt_per_pos += bsz

        if omniglot_first:  # ID 0 first
            bool_correct_pred0 = bool_correct_pred[:val_len0]
            bool_correct_pred1 = bool_correct_pred[val_len0:]
        else:
            bool_correct_pred1 = bool_correct_pred[:val_len1]
            bool_correct_pred0 = bool_correct_pred[val_len1:]

        task_running_correct[0] += bool_correct_pred0.sum().item()
        task_running_correct[1] += bool_correct_pred1.sum().item()

        assert val_bsz0 == val_bsz1
        assert val_len0 == val_len1
        counts += val_bsz0 * val_len0  # same size

        val_targets0 = val_targets0.transpose(0, 1)
        val_targets1 = val_targets1.transpose(0, 1)

        bool_correct_pred0 = bool_correct_pred0.transpose(0, 1)
        bool_correct_pred1 = bool_correct_pred1.transpose(0, 1)

        for b in range(bsz):
            # task 0
            prev_cl_end = 0
            _, cnts_uniq = torch.unique(
                val_targets0[b], sorted=True, return_counts=True)
            _, indices = torch.sort(val_targets0[b], stable=True)
            for cl in range(n_way):
                cl_cnts = cnts_uniq[cl]
                cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                cl_indices_len = len(cl_indices)
                prev_cl_end += cl_cnts

                for shot in range(k_shot):
                    if cl_indices_len > shot:
                        acc_per_shot[0][shot] += (
                            bool_correct_pred0[b][cl_indices[shot]].item())
                        cnt_per_shot[0][shot] += 1
            # task 1
            prev_cl_end = 0
            _, cnts_uniq = torch.unique(
                val_targets1[b], sorted=True, return_counts=True)
            _, indices = torch.sort(val_targets1[b], stable=True)
            for cl in range(n_way):
                cl_cnts = cnts_uniq[cl]
                cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                cl_indices_len = len(cl_indices)
                prev_cl_end += cl_cnts

                for shot in range(k_shot):
                    if cl_indices_len > shot:
                        acc_per_shot[1][shot] += (
                            bool_correct_pred1[b][cl_indices[shot]].item())
                        cnt_per_shot[1][shot] += 1

        if batch_id > num_steps:
            break

    running_correct = 100 * running_correct / running_total
    task_running_correct[0] = 100 * task_running_correct[0] / counts
    task_running_correct[1] = 100 * task_running_correct[1] / counts

    for key in acc_per_shot.keys():
        for shot in range(k_shot):
            shot_acc = (
                100 * acc_per_shot[key][shot] / cnt_per_shot[key][shot]
            )
            acc_per_shot[key][shot] = shot_acc

    # per position:
    for pos in range(k_shot * n_way * 2):
        acc_per_pos[pos] = 100 * acc_per_pos[pos] / cnt_per_pos

    return running_correct, task_running_correct, acc_per_shot, acc_per_pos
