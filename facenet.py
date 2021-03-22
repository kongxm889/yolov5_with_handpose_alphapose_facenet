import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.datasets import *

save_video = True
save_video_fource = 'mp4v'
save_video_path = 'result.mp4'
save_video_fps = 10.0
save_video_size = (1280, 720)

yolo_input_size = 640
min_person_size = 200
face_input_size_w = 200
face_input_size_h = 170
face_prebox_rate = 0.6

face2name_config = 1
draw_yolo_config = 0.7

tl = 10
color = (255, 191, 0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

class Facenet:
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.workers = 0 if os.name == 'nt' else 4
        self.faces_load()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((250,200)),      # (256, 256) 区别
            # transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

    def collate_fn(self, x):
        return x[0]

    def faces_load(self):
        self.dataset = datasets.ImageFolder('test_images_723')
        self.dataset.idx_to_class = {i: c for c, i in self.dataset.class_to_idx.items()}
        self.loader = DataLoader(self.dataset, collate_fn=self.collate_fn, num_workers=self.workers)

        aligned = []
        names = []
        for x, y in self.loader:
            x_aligneds, probs, boxes = self.mtcnn(x, return_prob=True)
            if x_aligneds is not None:
                print('Face detected with probability: {:6f}'.format(probs[0]))
                aligned.append(x_aligneds[0])
                names.append(self.dataset.idx_to_class[y])
        print(" ".join(names))
        aligned = torch.stack(aligned).to(device)
        self.embeddings = self.resnet(aligned).detach().cpu()

    def face_infer(self, frame):
        with torch.no_grad():
            faces, probs, boxes = self.mtcnn(frame, return_prob=True)
        return probs, boxes, faces