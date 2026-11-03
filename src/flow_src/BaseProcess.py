import os, json
from pathlib import Path
class BaseProcessClass:
	def __init__(self, trigger_path:Path, main_path:Path, global_obj:dict):
		self.trigger_path = trigger_path
		self.main_path = main_path
		self.global_obj = global_obj

		self.input_args = {}
		with open(f"{self.main_path}/inputs/{self.__class__.__name__}.json", "r") as f:
			self.input_args.update(json.load(f))
	def run(self):
		raise NotImplementedError("Subclasses must implement the run method.")