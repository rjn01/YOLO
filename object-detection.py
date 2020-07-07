import cv2
import numpy as np

#load yolo 
model_weight = "/Users/shiva/Desktop/yolo/yolov3.weights"
model_cfg = "/Users/shiva/Desktop/yolo/yolov3.cfg"
net = cv2.dnn.readNet(model_weight,model_cfg)
#we are using tiny yolo intead of yolo


#classes = ["bicycle","car","motorbike","truck","bus","train"]
classes=[]
with open("coco-labels","r") as f:
	classes = [line.strip() for line in f.readlines()]

#print(classes)

layers_name = net.getLayerNames()
output_layers = [layers_name[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#loading image
img = cv2.imread("pic.jpg")
#img = cv2.resize(img,None,fx=0.7,fy=0.7)
height,width,channels = img.shape

#detecting image
blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

#showing info on screen
class_ids=[]
confidences=[]
boxes=[]

for out in outs:
	for detection in out:
		scores = detection[5:]
		class_id = np.argmax(scores)
		confidence = scores[class_id]
		if confidence>0.3:
			#object detected
			center_x = int(detection[0]*width)
			center_y = int(detection[1]*height)
			w = int(detection[2]*width)
			h = int(detection[3]*height)
			x = int(center_x - w/2)
			y = int(center_y - h/2)

			boxes.append([x,y,w,h])
			confidences.append(float(confidence))
			class_ids.append(class_id)

indexes = cv2.dnn.	NMSBoxes(boxes, confidences, 0.5, 0.4)
#this line will remove the bounding boxes which are around same object

font = cv2.FONT_HERSHEY_SIMPLEX 
for i in range(len(boxes)):
	if i in indexes:   #show the boxes only if it is present in indexes
		x,y,w,h = boxes[i]
		label = str(classes[class_ids[i]])
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),1)
		cv2.putText(img,label,(x,y+10),font,0.5,(0,0,255),2)

cv2.imshow("img",img)


cv2.waitKey(0)
cv2.destroyAllWindows()
