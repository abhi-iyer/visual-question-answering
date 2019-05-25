import random
import json
from skimage import *

class Loader()
	def __init__(self):
		pass

	def load_image(path, size=224):
		img = skimage.io.imread(path)
		
		if len(img.shape) == 2:
			img = skimage.color.gray2rgb(img)
		
		size = min(img.shape[:2])
		y_end = int((img.shape[0] - size) / 2)
		x_end = int((img.shape[1] - size) / 2)

		croppedImage = img[y_end:y_end + size, x_end:x_end + size]
		final = skimage.transform.resize(croppedImage, (size, size))
		
		return final


	def get_vqa_data(is_train, sampling_ratio):
		if is_train:
			##replace with whatever json file we have
			annotations = json.load(open('nigg.json'))['annotations']
			questions = json.load(open('data/questions.json'))['questions']
			datasetPath = 'data/train2014/COCO_train2014_'
		else:
			annotations = json.load(open('data/mscoco_val2014_annotations.json'))['annotations']
			questions = json.load(open('data/OpenEnded_mscoco_val2014_questions.json'))['questions']
			datasetPath = 'data/val2014/COCO_val2014_'
		
		VQASample = list()
		
		for question, annotation in zip(questions, annotations):
			if question['question_id'] != annotation['question_id']:
				raise AssertionError("Non matching IDs")
			
			q = question['question']
			imageIndex = str(question['image_id'])
			imagePath = datasetPath
			
			for i in range(12 - len(imageIndex)):
				imagePath += '0'
			
			imagePath += imageIndex + '.jpg'
			VQASample.append((q, annotation['multiple_choice_answer'], imagePath))
		
		if sampling_ratio < 1:
			VQASample = random.sample(VQASample, int(round(len(VQASample) * sampling_ratio)))
		return VQASample

	def getAnswers(mode):
		if mode == 'training':
			annotations = json.load(open('nigg.json'))['annotations']
			questions = json.load(open('data/questions.json'))['questions']
		
		else:
			annotations = json.load(open('data/mscoco_val2014_annotations.json'))['annotations']
			questions = json.load(open('data/OpenEnded_mscoco_val2014_questions.json'))['questions']
		
		answers = dict()
		
		for question, annotation in zip(questions, annotations):
			if question['question_id'] != annotation['question_id']:
				raise AssertionError("Non matching ID")
			
			q = question['question']
			
			if q not in answers:
				answers[q] = set()
			
			for ans in annotation['answers']:
				answers[q].add(ans['answer'])
			
			answers[q].add(annotation['multiple_choice_answer'])
		return answers