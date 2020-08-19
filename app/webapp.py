import os,json,base64,torch,cv2
import numpy as np
from flask import Flask , render_template , request,session
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import GRACE,JPEG16

imgPath = "static/img/org.jpg"
desPath = "static/img/compress.jpg"
GRACE = GRACE.GRACE(scale=224)


# def predict_class(imgPath):
def predict_class(imgPath,model):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	img = Image.open(imgPath)
	trans = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	image = trans(img)
	image.unsqueeze_(0)
	image= image.to(device)
	nnmodel = models.resnet18(pretrained=False)
	if model=='resnet':
		nnmodel = models.resnet18(pretrained=False)
		nnmodel.load_state_dict(torch.load('dataset/resnet18-epoch100-acc70.316.pth'))
	elif model=='alexnet':
		nnmodel = models.alexnet(pretrained=False)
		nnmodel.load_state_dict(torch.load('dataset/alexnet-owt-4df8aa71.pth'))
	elif model=='vgg':
		nnmodel = models.vgg11(pretrained=False)
		nnmodel.load_state_dict(torch.load('dataset/vgg11-bbd30ac9.pth'))
	elif model== 'squeezenet':
		nnmodel = models.squeezenet1_0(pretrained=False)
		nnmodel.load_state_dict(torch.load('dataset/squeezenet1_0-a815701f.pth'))
	nnmodel = torch.nn.DataParallel(nnmodel).to(device)

	nnmodel.eval()
	out = nnmodel(image)
	_, indices = torch.sort(out, descending=True)
	result = indices.detach().cpu().numpy()
	return result
	



app = Flask(__name__)


@app.route('/')
def index():
	return render_template("index.html")

@app.route('/generateP',methods = ['POST','GET'])
def generateP():
	if request.method =='POST':
		nnmodel = request.form['nnmodel']
		ds = GRACE.get_ds()
		dx,img_rgb = GRACE.get_dx(datasetdir='dataset',model=nnmodel)
		gradients = GRACE.compute_gradients(dx,ds) 
		print("生成WR,WG,WB……")
		W_R, W_G, W_B = GRACE.get_W(gradients)
		gY, gV, gU = GRACE.compute_gYUV(gradients,W_R, W_G, W_B)
		SY, SU, SV = GRACE.get_Syuv(img_rgb,W_R, W_G, W_B)
		print("生成TY,TU,TV……")
		T_Y = GRACE.compute_T(gY,SY,B=0.00015)
		T_U = GRACE.compute_T(gU,SU,B=0.00015)
		T_V = GRACE.compute_T(gV,SV,B=0.00015)
		WR = W_R.item()
		WG = W_G.item()
		WB = W_B.item()
		TY = T_Y.detach().numpy().reshape(-1)
		TU = T_U.detach().numpy().reshape(-1)
		TV = T_V.detach().numpy().reshape(-1)

		f = open('P.txt', 'w')       #清空文件内容再写
		f.write(nnmodel);f.write('\n')
		f.write(str(WR));f.write('\n')
		f.write(str(WG));f.write('\n') 
		f.write(str(WB));f.write('\n')
		f.close()
		np.savetxt("TY.txt", TY ,fmt='%.28f',delimiter=',')
		np.savetxt("TU.txt", TU ,fmt='%.28f',delimiter=',')
		np.savetxt("TV.txt", TV ,fmt='%.28f',delimiter=',')

		return '%s %s %s %s' %(WR,WG,WB,TY)
	else:
		model = request.args.get('nnmodel')
		return 'success! get %s' %model


@app.route('/compressimg',methods = ['POST','GET'])
def compress():
	if request.method =='POST':
		f = open('P.txt','r',encoding='utf-8') #如果文件不是uft-8编码方式，读取文件可能报错
		f.readline()
		WR = float(f.readline())
		WG = float(f.readline())
		WB = float(f.readline())
		f.close()
		TY= np.loadtxt('TY.txt',delimiter=',')
		TU= np.loadtxt('TU.txt',delimiter=',')
		TV= np.loadtxt('TV.txt',delimiter=',')
		img = request.form['imgMsg']
		data = json.loads(img)
		for img_data in data:
			img_base64 = str(img_data['base64']);
		img_base64= img_base64.replace("data:image/jpeg;base64,","");
		fh = open(imgPath,"wb")
		fh.write(base64.b64decode(img_base64))
		fh.close();
		kjpeg = JPEG16.KJPEG(224, WR, WG, WB, TY, TU, TV)
		y_code, u_code, v_code = kjpeg.Compress(imgPath)
		img = kjpeg.Decompress(y_code, u_code, v_code)
		img.save(desPath, "jpeg")
		fh = open(desPath,"rb")
		base64_data = base64.b64encode(fh.read())
		s = base64_data.decode()
		return 'data:image/jpeg;base64,%s' %s
	else:
		img = request.args.get('imgMsg')
		return 'success! %s' %img

@app.route('/predictimg',methods = ['POST','GET'])
def predictimg():
	f = open('P.txt','r',encoding='utf-8') #如果文件不是uft-8编码方式，读取文件可能报错
	model = f.readline()
	model = model.strip()
	if request.method =='POST':
		return 'POST!'
	else:
		result = predict_class(desPath,model)
		return "%s %s %s %s %s %s" %(result[0,0],result[0,0],result[0,1],result[0,2],result[0,3],result[0,4])


if __name__ == '__main__':
	app.run(debug = True) 