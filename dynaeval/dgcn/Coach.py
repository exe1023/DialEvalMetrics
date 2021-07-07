import copy
import time
import os
import torch
from tqdm import tqdm
from sklearn import metrics
import dgcn
from torch import optim

log = dgcn.utils.get_logger()


class Coach:

    def __init__(self, trainset, devset, testset, model, opt, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.args = args
        self.best_dev_acc = None
        self.best_epoch = None
        self.best_state = None

    def load_ckpt(self, ckpt):
        self.best_dev_acc = ckpt["best_dev_acc"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)
        log.info("loaded pretrained model")
    
    def save_ckpt(self, ckpt):
        save_path = os.path.join(self.args.model_save_path, "best.pt")
        torch.save(ckpt, save_path)

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_dev_acc, best_epoch, best_state = self.best_dev_acc, self.best_epoch, self.best_state
        scheduler = optim.lr_scheduler.StepLR(self.opt.optimizer, step_size=1, gamma=0.5)
        # Train
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            ckpt = {"best_dev_acc": 0.0,
                    "best_epoch": epoch,
                    "best_state": copy.deepcopy(self.model.state_dict())}
            save_path = os.path.join(self.args.model_save_path, "epoch-{:02d}.pt".format(epoch))
            torch.save(ckpt, save_path)
            dev_acc = self.evaluate()
            log.info("[Dev set] [ACC {:.4f}]".format(dev_acc))
            if best_dev_acc is None or dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                log.info("Save the best model.")
                self.best_dev_acc = best_dev_acc
                self.best_epoch = best_epoch
                self.best_state = best_state
                ckpt = {"best_dev_acc": best_dev_acc,
                        "best_epoch": best_epoch,
                        "best_state": best_state}
                self.save_ckpt(ckpt)

            scheduler.step()   
        test_f1 = self.evaluate(test=True)
        log.info("[Test set] [f1 {:.4f}]".format(test_f1))

        # The best
        self.model.load_state_dict(best_state)
        log.info("")
        log.info("Best in epoch {}:".format(best_epoch))
        dev_acc = self.evaluate()
        log.info("[Dev set] [f1 {:.4f}]".format(dev_acc))
        test_f1 = self.evaluate(test=True)
        log.info("[Test set] [f1 {:.4f}]".format(test_f1))

    def train_epoch(self, epoch):

        start_time = time.time()
        epoch_loss = 0
        self.model.train()
        self.trainset.shuffle()
        for idx in range(len(self.trainset)):
            self.model.zero_grad()
            data = self.trainset[idx]

            for k, v in data.items():
                data[k] = v.to(self.args.device)

            coh_loss = self.model.get_loss(data)

            epoch_loss += coh_loss.item()
            coh_loss.backward()
            self.opt.step()
            log.info("[Epoch %d] [Step %d] [Coh Loss: %f]" % (epoch, idx, epoch_loss / (idx+1)))

        end_time = time.time()
        log.info("")
        log.info("[Epoch %d] [Loss: %f] [Time: %f]" %
                 (epoch, epoch_loss, end_time - start_time))

    def evaluate(self, test=False):
        dataset = self.testset if test else self.devset
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    data[k] = v.to(self.args.device)
                rst = self.model(data)
                y_hat = rst[0]
                preds.append(y_hat.detach().to("cpu"))

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()

            acc = metrics.accuracy_score(golds, preds)

            return acc


