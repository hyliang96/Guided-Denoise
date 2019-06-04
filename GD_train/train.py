from tqdm import tqdm
import sys
import numpy as np
import time
import torch
from torch.autograd import Variable
from torch import optim
from torch.nn import DataParallel
# from print_utlis import progess_bar
import json

def tablecell(head,data,leng,form,ifhead,seg='  '):
    if ifhead:
        return ('{:>%s}'%leng).format(head)+seg
    elif data == '':
        return ('{:>%s}'%leng).format('')+seg
    else:
        return ('{:>%s%s}'%(leng,form)).format(data)+seg

def output_state(state='', epoch='', orig_acc='', adv_acc='',
          dt='', time_per_batch='', time_per_img='',
          lr='', loss='', ifhead=False):

    line =\
        tablecell('state',state,10,'s', ifhead)+\
        tablecell('epoch',epoch,10,'d', ifhead)+\
        tablecell('orig_acc',orig_acc*100,10,'.3f', ifhead)+\
        tablecell('adv_acc',adv_acc*100,10,'.3f', ifhead)+\
        tablecell('all_time/s',dt, 10,'.1f', ifhead)+\
        tablecell('tbatch/ms',time_per_batch, 10, '.1f', ifhead)+\
        tablecell('timg/ms',time_per_img*1000, 10,'.3f', ifhead)+\
        tablecell('lr',lr,10,'.7f', ifhead)

    if ifhead:
        line+='loss per level'
    elif type(loss)==str and loss=='':
        line+=""
    else:
        line+=''.join(['%.5f  ' % loss_i for loss_i in loss])

    print(line)

def train(epoch, net, data_loader, optimizer, get_lr, loss_idcs = [4], requires_control = True):
    start_time = time.time()
    net.eval()
    if isinstance(net, DataParallel):
        net.module.net.denoise.train()
    else:
        net.net.denoise.train()

    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    orig_acc = []
    acc = []
    loss = []
    if requires_control:
        control_acc = []
        control_loss = []

    n_batch = len(data_loader)
    pbar = tqdm(total=n_batch, leave=False, position=0)     
    for i, (orig, adv, label) in enumerate(data_loader):
        prefix='epoch %03d train' % epoch
        suffix="Batch %4d / %4d"%(i+1, n_batch)
        pbar.set_description_str(prefix)
        pbar.set_postfix_str(suffix)
        pbar.update()

        # progess_bar((i+1)/n_batch, precision=4,num_block=70, prefix='epoch %03d train' % epoch,
        #     suffix="Batch %4d / %4d"%(i+1, n_batch), clean=True)
        # orig = Variable(orig.cuda(async = True), volatile = True)
        # adv = Variable(adv.cuda(async = True), volatile = True)
        orig = Variable(orig.cuda(async = True))
        adv = Variable(adv.cuda(async = True))

        if not requires_control:
            orig_pred, adv_pred, l = net(orig, adv, requires_control = False)
        else:
            orig_pred, adv_pred, l, control_pred, cl = net(orig, adv, requires_control = True)

        _, idcs = orig_pred.data.cpu().max(1)
        orig_acc.append(float(torch.sum(idcs == label)) / len(label))
        _, idcs = adv_pred.data.cpu().max(1)
        acc.append(float(torch.sum(idcs == label)) / len(label))
        total_loss = 0
        for idx in loss_idcs:
            total_loss += l[idx].mean()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        loss_values = []
        for ll in l:
            loss_values.append(ll.mean().item())
        loss.append(loss_values)

        if requires_control:
            _, idcs = control_pred.data.cpu().max(1)
            control_acc.append(float(torch.sum(idcs == label)) / len(label))
            loss_values = []
            for ll in cl:
                loss_values.append(ll.mean().item())
            control_loss.append(loss_values)

    pbar.close()
        #print('\torig_acc %.3f, acc %.3f, control_acc %.3f' % (
        #    orig_acc[-1], acc[-1], control_acc[-1]))
        #print('\tloss: %.5f, %.5f, %.5f, %.5f, %.5f' % (
        #    loss[-1][0], loss[-1][1], loss[-1][2], loss[-1][3], loss[-1][4]))
        #if requires_control:
        #    print('\tloss: %.5f, %.5f, %.5f, %.5f, %.5f' % (
        #        control_loss[-1][0], control_loss[-1][1], control_loss[-1][2], control_loss[-1][3], control_loss[-1][4]))
        #print

    orig_acc = np.mean(orig_acc)
    acc = np.mean(acc)
    loss = np.mean(loss, 0)

    if requires_control:
        control_acc = np.mean(control_acc)
        control_loss = np.mean(control_loss, 0)
    end_time = time.time()
    dt = end_time - start_time

    n_go = 3 if requires_control else 2
    time_per_batch = dt/(n_batch * n_go)
    time_per_img = dt/(data_loader.batch_size * n_batch *  n_go)

    output_state(state='train', epoch=epoch, orig_acc=orig_acc, adv_acc=acc,
                 dt=dt, time_per_batch=time_per_batch, time_per_img=time_per_img,lr=lr, loss=loss)
    if requires_control:
        output_state(state=' - defense', adv_acc=control_acc, loss=control_loss)


    # if requires_control:
    #     print('Epoch %3d (lr %.5f): orig_acc %.3f, acc %.3f, control_acc %.3f, time %3.1f' % (
    #         epoch, lr, orig_acc, acc, control_acc, dt))
    # else:
    #     print('Epoch %3d (lr %.5f): orig_acc %.3f, acc %.3f, time %3.1f' % (
    #         epoch, lr, orig_acc, acc, dt))

    # print('\tloss: %.5f, %.5f, %.5f' % (
    #     loss[0], loss[1], loss[2]))

    # if requires_control:
    #     print('\tloss: %.5f, %.5f, %.5f' % (
    #         control_loss[0], control_loss[1], control_loss[2]))
    # print

def val(epoch, net, data_loader, requires_control = True):
    start_time = time.time()
    net.eval()

    orig_acc = []
    acc = []
    loss = []
    if requires_control:
        control_acc = []
        control_loss = []

    n_batch = len(data_loader)
    pbar = tqdm(total=n_batch, leave=False, position=0)     

    for i, (orig, adv, label) in enumerate(data_loader):
        prefix='epoch %03d val  ' % epoch
        suffix="Batch %4d / %4d"%(i+1, n_batch)
        pbar.set_description_str(prefix)
        pbar.set_postfix_str(suffix)
        pbar.update()

        # progess_bar((i+1)/n_batch, precision=4, num_block=70, prefix='epoch %03d val  ' % epoch,
        #     suffix="Batch %4d / %4d"%(i+1, n_batch), clean=True)

        # orig = Variable(orig.cuda(async = True), volatile = True)
        # adv = Variable(adv.cuda(async = True), volatile = True)
        orig = Variable(orig.cuda(async = True))
        adv = Variable(adv.cuda(async = True))

        if not requires_control:
            orig_pred, adv_pred, l = net(orig, adv, requires_control = False, train = False)
        else:
            orig_pred, adv_pred, l, control_pred, cl = net(orig, adv, requires_control = True, train = False)

        _, idcs = orig_pred.data.cpu().max(1)
        orig_acc.append(float(torch.sum(idcs == label)) / len(label))
        _, idcs = adv_pred.data.cpu().max(1)
        acc.append(float(torch.sum(idcs == label)) / len(label))
        loss_values = []
        for ll in l:
            loss_values.append(ll.mean().item())
        loss.append(loss_values)

        if requires_control:
            _, idcs = control_pred.data.cpu().max(1)
            control_acc.append(float(torch.sum(idcs == label)) / len(label))
            loss_values = []
            for ll in cl:
                loss_values.append(ll.mean().item())
            control_loss.append(loss_values)

        #print('\torig_acc %.3f, acc %.3f, control_acc %.3f' % (
        #    orig_acc[-1], acc[-1], control_acc[-1]))
        #print('\tloss: %.5f, %.5f, %.5f, %.5f, %.5f' % (
        #    loss[-1][0], loss[-1][1], loss[-1][2], loss[-1][3], loss[-1][4]))
        #if requires_control:
        #    print('\tloss: %.5f, %.5f, %.5f, %.5f, %.5f' % (
        #        control_loss[-1][0], control_loss[-1][1], control_loss[-1][2], control_loss[-1][3], control_loss[-1][4]))
        #print
    pbar.close()    

    orig_acc = np.mean(orig_acc)
    acc = np.mean(acc)
    loss = np.mean(loss, 0)
    if requires_control:
        control_acc = np.mean(control_acc)
        control_loss = np.mean(control_loss, 0)
    end_time = time.time()
    dt = end_time - start_time


    n_go = 3 if requires_control else 2
    time_per_batch = dt/(n_batch * n_go)
    time_per_img = dt/(data_loader.batch_size * n_batch *  n_go)


    output_state(state='val', orig_acc=orig_acc, adv_acc=acc,
                 dt=dt, time_per_batch=time_per_batch, time_per_img=time_per_img, loss=loss)
    if requires_control:
        output_state(state=' - defense', adv_acc=control_acc, loss=control_loss)


    # if requires_control:
    #     print('Validation: orig_acc %.3f, acc %.3f, control_acc %.3f, time %3.1f' % (
    #         orig_acc, acc, control_acc, dt))
    # else:
    #     print('Validation: orig_acc %.3f, acc %.3f, time %3.1f' % (
    #         orig_acc, acc, dt))

    # print('\tloss: %.5f, %.5f, %.5f' % (
    #     loss[0], loss[1], loss[2]))

    # if requires_control:
    #     print('\tloss: %.5f, %.5f, %.5f' % (
    #         control_loss[0], control_loss[1], control_loss[2]))
    # print
    # print


class TagAccuracy(object):
    def __init__(self):
        self.sum = {}
        self.n = {}
        self._mean = {}

    def add(self, tags, pred, label):
        # pred []
        _, idcs = pred[-1].data.cpu().max(1)
        corrects = idcs == label

        for correct, tag in zip(corrects, tags):
            if tag in self.sum.keys():
                self.sum[tag] += correct.item()
                self.n[tag] += 1
            else:
                self.sum[tag] = correct.item()
                self.n[tag] = 1

            self._mean[tag] = self.sum[tag]/self.n[tag] 
        
    def mean(self):
        return self._mean

    def keys(self):
        return self.sum.keys()

def test(net, data_loader, result_file_name, defense = True):
    # start_time = time.time()
    net.eval()

    # # defense
    # n_acc_by_attack = {}
    # acc_by_attack = {}
    # # nodefense
    # n_acc_by_attack_nodf = {}
    # acc_by_attack_nodf = {}

    # n_data = 0
    if defense:
        acc = TagAccuracy()
    acc_nodf = TagAccuracy()
    acc_pbars={}
    attack_list = []

    n_batch = len(data_loader)
    # n_batch=10
    pbar = tqdm(total=n_batch, leave=True, position=0) 
    
    for i, (adv, label, attacks) in enumerate(data_loader):
        # adv = Variable(adv.cuda(async = True), volatile = True)
        adv = Variable(adv.cuda(async = True))

        if defense:
            adv_pred = net(adv, defense = True)
            acc.add(attacks, adv_pred, label)
            # _, idcs = adv_pred[-1].data.cpu().max(1)
            # corrects = idcs == label
        adv_pred_nodf = net(adv, defense = False)
        acc_nodf.add(attacks, adv_pred_nodf, label)

        prefix="Test on adv | Batch %4d/%4d"%(i+1, n_batch)
        acc_output = 'defense/no_defense: ' if defense else 'no_defnese: '
        acc_output += ' '.join(
                [ '%s:%.3f/%.3f' % (attack, acc.mean()[attack], acc_nodf.mean()[attack]) 
                    for attack in acc.keys()]
            )
        # progess_bar((i+1)/n_batch, precision=4, num_block=50, 
        #     prefix=prefix,
        #     suffix=acc_output, 
        #     clean=True)
        pbar.set_description_str(prefix)
        # pbar.set_postfix_str(acc_output)
        pbar.update()

        for attack in attacks:
            if not attack in acc_pbars.keys():
                acc_pbars[attack] = tqdm(total=data_loader.num_orig_data, leave=True, position=1+len(attack_list))
                attack_list.append(attack)
            if defense:
                acc_attck = acc.mean()[attack]
                acc_nodf_attck = acc_nodf.mean()[attack]
                if acc_nodf_attck != 0:
                    acc_rate = (acc_attck-acc_nodf_attck)/acc_nodf_attck
                else:
                    acc_rate = float('nan')
                acc_str = 'defense acc: %.3f / no_defense acc: %.3f, higher %6.3f' % (acc_attck, acc_nodf_attck, acc_rate)
            else:
                acc_nodf_attck = acc_nodf.mean()[attack]
                acc_str = 'no_defense acc: %.3f, higher %.3f' % acc_nodf_attck
            acc_pbars[attack].set_description_str(attack+': ')
            acc_pbars[attack].set_postfix_str(acc_str)
            acc_pbars[attack].update()

        # if i == 10:
        #     break
            
    pbar.close()
    for attack in attack_list:
        acc_pbars[attack].close()

        # for correct, attack in zip(corrects, attacks):
        #     n_data += 1
        #     if attack in n_acc_by_attack.keys():
        #         n_acc_by_attack[attack] += correct.item()
        #     else:
        #         n_acc_by_attack[attack] = correct.item()
        #     acc_by_attack[attack] = n_acc_by_attack[attack]/n_data

    for i in range(len(attack_list)+1):
        print()
    print()    

    if defense:
        acc_rates = {}
        for attack in acc.keys():
            acc_attck = acc.mean()[attack]
            acc_nodf_attck = acc_nodf.mean()[attack]
            acc_rate = (acc_attck-acc_nodf_attck)/acc_nodf_attck
            acc_rates[attack] = acc_rate

    # print('Test | acc under different transferred attack')
    # if defense:
    #     print('defense:    ', end='')
    #     print(acc.mean())
    # print('no defense: ', end='')
    # print(acc_nodf.mean())
    # if defense:
    #     print('higher:     ', end='')
    #     print(acc_rates)

    if defense:
        log_content={}
        for attack in acc.keys():
            log_content[attack]={'defense:': acc.mean()[attack], 
                                'no_defense:': acc_nodf.mean()[attack],
                                'rate': acc_rates[attack]
                                }
    else:
        log_content = acc_nodf.mean()

    print('transferred attack method: test accuracy')
    for key,value in log_content.items():
        print(key,':',value)

    with open(result_file_name+'.json', 'w') as f:
        json.dump(log_content, f, ensure_ascii=False)


class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass
