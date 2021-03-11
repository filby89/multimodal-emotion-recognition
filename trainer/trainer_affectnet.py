import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, make_barplot, features_blobs, setup_cam, returnCAM
import matplotlib as mpl
import random

mpl.use('Agg')
import matplotlib.pyplot as plt
import model.metric

class Trainer(BaseTrainer):
	"""
	Trainer class
	"""
	def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
				 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
		super().__init__(model, criterion, metric_ftns, optimizer, config)
		self.data_loader = data_loader

		if len_epoch is None:
			# epoch-based training
			self.len_epoch = len(self.data_loader)
		else:
			# iteration-based training
			self.data_loader = inf_loop(data_loader)
			self.len_epoch = len_epoch

		self.valid_data_loader = valid_data_loader
		self.do_validation = self.valid_data_loader is not None
		self.lr_scheduler = lr_scheduler
		self.log_step = 50

		self.criterion_categorical = criterion

		self.train_metrics = MetricTracker('accuracy', 'balanced_accuracy', 'loss', writer=self.writer)
		self.valid_metrics = MetricTracker('accuracy', 'balanced_accuracy', 'loss', writer=self.writer)

	def _train_epoch(self, epoch, phase="train"):
		"""
		Training logic for an epoch

		:param epoch: Integer, current training epoch.
		:return: A log that contains average loss and metric in this epoch.
		"""

		print("Finding LR")
		for param_group in self.optimizer.param_groups:
			print(param_group['lr'])

		if phase == "train":
			self.model.train()
			self.train_metrics.reset()
			torch.set_grad_enabled(True)
			metrics = self.train_metrics
		elif phase == "val":
			self.model.eval()
			self.valid_metrics.reset()
			torch.set_grad_enabled(False)
			metrics = self.valid_metrics

		outputs = []
		targets = []

		data_loader = self.data_loader if phase == "train" else self.valid_data_loader

		for batch_idx, (images, target, ages) in enumerate(data_loader):

			images, target, ages = images.to(self.device), target.to(self.device), ages.to(self.device)

			if phase == "train":
				self.optimizer.zero_grad()

			out = self.model(images)

			loss = self.criterion_categorical(out['categorical'], target)

			if phase == "train":
				loss.backward()
				# torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=10.0)
				self.optimizer.step()

			output = out['categorical'].cpu().detach().numpy()
			target = target.cpu().detach().numpy()
			outputs.append(output)
			targets.append(target)

			if batch_idx % self.log_step == 0:
				self.logger.debug('{} Epoch: {} {} Loss: {:.6f}'.format(
					phase,
					epoch,
					self._progress(batch_idx, data_loader),
					loss.item()))

			if batch_idx == self.len_epoch:
				break

		if phase == "train":
			self.writer.set_step(epoch)
		else:
			self.writer.set_step(epoch, "valid")

		metrics.update('loss', loss.item())

		output = np.concatenate(outputs, axis=0)
		target = np.concatenate(targets, axis=0)

		accuracy = model.metric.accuracy(output, target)
		metrics.update("accuracy", accuracy)

		balanced_accuracy = model.metric.balanced_accuracy(output, target)
		metrics.update("balanced_accuracy", balanced_accuracy)

		log = metrics.result()

		if phase == "train":
			if self.do_validation:
				val_log = self._train_epoch(epoch, phase="val")
				log.update(**{'val_' + k: v for k, v in val_log.items()})

			return log

		elif phase == "val":
			if self.lr_scheduler is not None:
				if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
					self.lr_scheduler.step(loss.item())
				else:
					self.lr_scheduler.step()

			self.writer.save_results(output, "output")
			self.writer.save_results(target, "target")

			return metrics.result()


	def _progress(self, batch_idx, data_loader):
		base = '[{}/{} ({:.0f}%)]'
		if hasattr(data_loader, 'n_samples'):
			current = batch_idx * self.data_loader.batch_size
			total = data_loader.n_samples
		else:
			current = batch_idx
			total = self.len_epoch
		return base.format(current, total, 100.0 * current / total)
