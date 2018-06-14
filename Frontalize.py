import os
import sys
import numpy as np
import cv2
import math
import dlib
import time
deepInpaint = True
from pypoi import poissonblending
try:
    import inpaint
except:
    print("Could Not Load deep inpainting module, will default to opencv...")
    deepInpaint = False
verbose=False
def detectLandmarks(img_bgr,detector,predictor,detscale = .3):
    if img_bgr is not None:
        img_bgr = cv2.resize(img_bgr, (int(img_bgr.shape[1] * .5), int(img_bgr.shape[0] * .5)))
        img_bgr_det = cv2.resize(img_bgr, (int(img_bgr.shape[1] * detscale), int(img_bgr.shape[0] * detscale)))
        img_det = cv2.cvtColor(img_bgr_det, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        dets = detector(img_det, 1)
        for k, ds in enumerate(dets):
            d = dlib.rectangle(int(float(ds.left()) / detscale), int(float(ds.top()) / detscale),
                               int(float(ds.right()) / detscale), int(float(ds.bottom()) / detscale))
            shape = predictor(img, d)
            landmarks = []
            for i in range(shape.num_parts):
                img_bgr = cv2.circle(img_bgr, (shape.part(i).x, shape.part(i).y), 2, (0, 0, 255), -1)
                landmarks.append([shape.part(i).x, shape.part(i).y])
            landmarks = np.asarray(landmarks, dtype='float32')
            chull = cv2.convexHull(landmarks)
            chullminx = chull[:, :, 0].min()
            chullminy = chull[:, :, 1].min()
            chullmaxx = chull[:, :, 0].max()
            chullmaxy = chull[:, :, 1].max()

            top = int(min(chullminy, d.top())) - 1
            bottom = int(max(chullmaxy, d.bottom())) + 1
            left = int(min(chullminx, d.left())) - 1
            right = int(max(chullmaxx, d.right())) + 1
            return ((left,top,right,bottom),landmarks,img)
    return (None,None,None)

def parRunImage(file,detector,predictor,frontalizer,outputDir):
    try:
        img_bgr = cv2.imread(file)
        frontal_raw, inpainted = runFrontalizationOnImage(img_bgr, detector, predictor, frontalizer)
        # cv2.imwrite(os.path.join(outputDir, 'inpaint_' + file), inpainted)
        cv2.imwrite(os.path.join(outputDir,os.path.basename(file)), frontal_raw)
    except:
        pass

def runImageDir(imgDir,outputDir,numcores=1):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    predictor_path = './data/shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    frontalizer = Frontalizer(scale=1)
    files = []
    count = 0
    for dir in os.listdir(imgDir):
        dirPath = os.path.join(imgDir,dir)
        if os.path.isdir(dirPath):
            for f in os.listdir(dirPath):
                file = os.path.join(dirPath,f)
                files.append(file)
                count += 1
                print(count)
        # for name in files:
        #     file = os.path.join(root,name)
        #     if not os.path.isdir(os.path.join(imgDir,file)) and not file.startswith('.'):
        #         files.append(file)

    from joblib import Parallel,delayed
    Parallel(n_jobs=numcores)(delayed(parRunImage)(file,detector,predictor,frontalizer,outputDir) for file in files)


def runFrontalizationOnImage(img_bgr,detector,predictor,frontalizer):
    faceRect, landmarks, img = detectLandmarks(img_bgr, detector, predictor)
    if faceRect is not None:
        # poseFrame = frontalizer.genPoseOutputImage(img, landmarks)
        img_bgr = cv2.rectangle(img_bgr, (faceRect[0], faceRect[1]), (faceRect[2], faceRect[3]), (255, 0, 0), 5)
        dispImg = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cv2.imshow('frame1', dispImg)
        # cv2.imshow('pose',poseFrame)
        fdata = frontalizer.frontalizeImage(img, landmarks)
        frontalFace = fdata[0]
        dgenMap = fdata[1]
        ff_bgr = cv2.cvtColor(frontalFace, cv2.COLOR_BGR2RGB)
        inpainted_bgr = None
        if dgenMap is not None and len(dgenMap) > 0:
            inpainted_bgr = cv2.cvtColor(dgenMap, cv2.COLOR_BGR2RGB)
        return(ff_bgr,inpainted_bgr)
    return (None,None)
def testFrontalize():
    img_bgr = cv2.imread('./data/testfaces/face5.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img2 = io.imread('face.jpg')

    predictor_path = './data/shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    # print("Processing file: {}".format(f))
    # win = dlib.image_window()

    # win.clear_overlay()
    # win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.

    frontalizer = Frontalizer(scale=1)

    cap = cv2.VideoCapture(0)
    # plt.ion()
    detscale = .3
    while (cap.isOpened()):
        t0 = time.time()
        ret, img_bgr = cap.read()
        frontal_raw,inpainted = runFrontalizationOnImage(img_bgr,detector,predictor,frontalizer)
        if frontal_raw is not None:
            cv2.imshow('raw',frontal_raw)
            if inpainted is not None:
                cv2.imshow('inpainted',inpainted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        t1 = time.time()
        print('framerate: ', 1/(t1-t0), 'FPS')
    cap.release()
    cv2.destroyAllWindows()


        #
        # win.add_overlay(dets)
        # dlib.hit_enter_to_continue()

def mirrorInpaint(image,mask):
    maskFlip = np.fliplr(mask)
    neededParts = np.fliplr(image*(maskFlip))
    return image*(1-mask)+neededParts

def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

class Model3d:
    model = []
    threedee = []
    landmarks = []
    width = -1
    height = -1
    refU = []
    outA = []
    render_dims = -1
    def __init__(self):
        self.model = []
        self.threedee = []
        self.landmarks = []
        self.width = -1
        self.height = -1
        self.refU = []
        self.outA = []
    def load3dModel(self,filepath):
        pass
class Triang:
    target = []
    targetZ = []
    xmin = -1
    xmax = -1
    ymin = -1
    ymax = -1
    ref = []
    coor = []
    def __init__(self):
        self.target = []
        self.targetZ = []
        self.xmin = -1
        self.xmax = -1
        self.ymax = -1
        self.ymax = -1
        self.ref = []
        self.coor = []
class Tri:
    mins = (-1,-1)
    maxs = (-1,-1)
    size = (-1,-1)
    inds = []
    points = []
    trianglePoints = []
    indices = []
    count = -1
    triangles = []
    listOfIndexesPerTriangle = {}
    def __init__(self):
        self.mins = (-1,-1)
        self.maxs = (-1,-1)
        self.inds = []
        self.points = []
        self.trianglePoints = []
        self.indices = []
        self.count = -1
        self.triangles = []
        self.triangleMap = []
        self.listOfIndexesPerTriangle = {}
    def create_tri_map(self,img):
        outImg = img.copy()
        size = img.shape
        r = (0, 0, size[1], size[0])
        count = 0
        for i in range(len(self.trianglePoints)):
            t = self.trianglePoints[i]
            pt1 = (t[0][0], t[0][1]);pt2 = (t[1][0], t[1][1]);pt3 = (t[2][0], t[2][1])
            a = np.asarray([[pt1[0],pt1[1]],[pt2[0],pt2[1]],[pt3[0],pt3[1]]])
            if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
                outImg = cv2.fillPoly(outImg, np.int32([a]), count)
            count+=1

        self.triangleMap=outImg
    def findTriangleIndex(self,point):
        return self.triangleMap[point[1],point[0]]
        # for i in range(len(self.trianglePoints)):
        # #     if PointInTriangle(point,self.trianglePoints[i][0],self.trianglePoints[i][1],self.trianglePoints[i][2]): return i
        # # return -1
        #     if cv2.pointPolygonTest(np.asarray(self.trianglePoints[i]),point,False) >= 0:
        #         knownind = i
        #         print(knownind,',',testind)
        #         return i
        # return -1
def sign(p1, p2, p3):
  return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def PointInAABB(pt, c1, c2):
  return c2[0] <= pt[0] <= c1[0] and \
         c2[1] <= pt[1] <= c1[1]

def PointInTriangle(pt,v1, v2, v3):
  b1 = sign(pt, v1, v2) <= 0
  b2 = sign(pt, v2, v3) <= 0
  b3 = sign(pt, v3, v1) <= 0

  return ((b1 == b2) and (b2 == b3)) and \
         PointInAABB(pt, map(max, v1, v2, v3), map(min, v1, v2, v3))


def countUnique(patch, x, y):
    return (x,y,len(np.unique(patch)))

class uniqueCounter:
    def __init__(self,windowSize,step,numCores = 5):
        self.degenerateMap = []
        self.replace_mask = []
        self.windowSize = windowSize
        self.step = step
        self.numcores = numCores
        # self.pool = Pool(numCores)
    def apply_result(self,result):
        x = result[0]
        y = result[1]
        v = result[2]
        self.degenerateMap[y:y+self.step,x:x+self.step] = v
    def buildDegenerateMap(self,mappedPixelLocs,final_mapped_img):
        # self.pool = Pool(self.numcores)
        originalStep = self.step
        self.degenerateMap = np.zeros(mappedPixelLocs.shape,dtype=np.int32)
        self.replace_mask = np.zeros_like(self.degenerateMap)
        # while self.step >= 2:
        for x in range(0,mappedPixelLocs.shape[1]-math.ceil(self.windowSize),self.step):
            for y in range(0,mappedPixelLocs.shape[0]-math.ceil(self.windowSize),self.step):
                centerx = x+int(self.windowSize/2)
                centery = y+int(self.windowSize/2)
                if np.sum(final_mapped_img[centery,centerx]) > 0 and self.replace_mask[centery,centerx] == 0:
                    window = mappedPixelLocs[y:y+self.windowSize,x:x+self.windowSize]
                    self.degenerateMap[centery:centery + self.step, centerx:centerx + self.step] = len(np.unique(window))
        self.degenerateMap = np.uint8((self.degenerateMap / self.degenerateMap.max()) * 255)
        ret, self.replace_mask = cv2.threshold(self.degenerateMap, 135, 255, cv2.THRESH_BINARY_INV)
        self.step = int(self.step/2)
            # cv2.imshow('deg',self.replace_mask)
            # cv2.imshow('rep', self.degenerateMap)
            # cv2.waitKey()
        self.step = originalStep
        # self.pool.apply_async(countUnique,(window,centerx,centery),callback=self.apply_result)
        # self.pool.close()
        # self.pool.join()
        return self.degenerateMap

class Frontalizer:
    model3d = Model3d()
    normX3d = []
    ptsxv = []
    pts = []
    ptsx = []
    img = []
    x3d = []
    homogen3D = []
    Z = []
    x3d_zm = []
    scale = []
    U = []
    homo_x3d = []
    normX3d = []
    modelLandmarks = []
    imgModel = []
    modelOuterMask = []
    degenerateBuilder = uniqueCounter(15,10,1)
    def __init__(self,scale=1,model3D=None,modelFile='./data/imgModel.csv',):
        self.img = []; self.normX3d = []; self.ptsxv = []; self.pts = []; self.ptsx = []; self.x3d = []
        self.homo_x3d = []; self.Z = []; self.Z_ext = []; self.x3d_zm=[]; self.scale = []; self.U = []; self.homo_x3d=[]; self.normX3d=[];
        self.modelLandmarks = []; self.imgModel = []
        self.model3d = model3D
        self.img = np.loadtxt(modelFile,delimiter=',')
        self.mask3D = []
        self.img = cv2.resize(self.img,(int(self.img.shape[1]*scale),int(self.img.shape[0]*scale)))
        imin = self.img.min()
        imax = self.img.max()
        self.img = (self.img-imin)/(imax-imin)*255
        self.generateDefaultLandmarkTemplate(scale=scale)
        self.pixelLocs = []
        self.modelOuterMask = None
        if deepInpaint:
            self.inPainter = inpaint.Inpaint('./data/latest')
    def estimatePoseParameters(self,landmarks,a,c,t):
        eye1Points = landmarks[9:13]
        eye2Points = landmarks[20:25]
        mouthPoints = landmarks[31:50]
        e1 = eye1Points.mean(axis=1)
        e2 = eye2Points.mean(axis=1)
        m = mouthPoints.mean(axis=1)
        cx = (e1[0]+e2[0]+m[0])/3
        cy = (e1[1]+e2[1]+m[1])/3
        P = np.asarray([[e1[0]-c[0],e1[1]-c[1]],[e2[0]-c[0],e2[1]-c[1]],[m[0]-c[0],m[1]-c[1]]])
        Sigma = 1/3*np.dot(P.transpose(),P)
        r=Sigma[0,0]+Sigma[1,1]+math.sqrt(math.pow(Sigma[0,0]-Sigma[1,1],2)+4*Sigma[0,1])
        pitch = math.asin()

    def frontalizeImage(self,faceImg,landmarks):
        bestImage = None
        chull = cv2.convexHull(landmarks)
        model3d = Model3d()

        if model3d is not None:
            if landmarks is not None and landmarks.shape[0] == 68:
                t0 = time.time()
                side = 0
                # cameraMatrices = self.estimateCamera(landmarks)
                side = self.simplePoseDetector(landmarks,faceImg,chull)
                x2d = landmarks
                x2d2 = landmarks[7:10]
                x2d = np.vstack((x2d2,x2d[17:68]))
                mua = cv2.reduce(x2d,0,cv2.REDUCE_SUM)
                muarep = cv2.repeat(mua,x2d.shape[0],1)
                x2d_zm = x2d-muarep
                powx2d_zm = np.square(x2d_zm)
                d_sq = cv2.reduce(powx2d_zm[:,0],1,cv2.REDUCE_SUM)
                mean_sq = d_sq.mean()
                s = math.sqrt(mean_sq/2)
                scalexyz = math.sqrt(powx2d_zm.sum()/(2*x2d.shape[0]))
                scale = np.asarray([1/scalexyz,1/scalexyz])
                T = np.eye(2)/scalexyz
                T = np.vstack((T,np.zeros((1,2),dtype='float32')))
                mscaletmp = -(scale*mua).transpose()
                mscaletmp = np.vstack((mscaletmp,np.ones((1,1),dtype='float32')))
                T = np.hstack((T,mscaletmp))
                homo_x2d = np.hstack((x2d,np.ones((x2d.shape[0],1),dtype='float32')))
                normx2d = np.dot(T,homo_x2d.transpose())
                zero_vec = np.zeros((1,4),dtype='float32')
                X_3D = np.zeros((136,8),dtype='float32')

                for ih in range(68):
                    i = ih*2
                    vec3D = np.zeros((1,4))
                    vec3D[0,:] = self.normX3d[:,ih].transpose()
                    newv = np.hstack((vec3D,zero_vec))
                    X_3D[i] = newv
                    newv = np.hstack((zero_vec,vec3D))
                    X_3D[i+1] = newv
                X_3D = np.vstack((X_3D[14:20,:],X_3D[34:136,:]))
                x_2D = normx2d[0:2,:].transpose()
                x_2D = x_2D.reshape((1,-1)).transpose()
                X_3Ds = X_3D
                x_2Ds = x_2D
                t1 = time.time()
                if verbose:
                    print("Setup coordinates time: ", t1-t0)
                t0 = time.time()
                retval, Ptv = cv2.solve(np.dot(X_3Ds.transpose(),X_3Ds),np.dot(X_3Ds.transpose(),np.float32(x_2Ds)),flags=cv2.DECOMP_NORMAL) #This is our final camera
                Pt = Ptv.transpose().reshape((2,-1))
                addon = np.zeros((1,4),dtype='float32')
                addon[0,3] = 1
                Pt = np.vstack((Pt,addon))
                ret, P = cv2.solve(T,np.float64(Pt),flags=cv2.DECOMP_NORMAL)
                P = np.dot(P,self.U)
                #Compute affine 3D to 2D camera
                x_proj = np.dot(P,self.homogen3D.transpose()) #here we see how good the camera is
                homogen3dal1 = np.zeros((71,4),dtype='float32')
                for i in range(71):
                    if i > self.pts.shape[0]: self.Z[i] = 0
                    else:
                        if i < self.Z.shape[0]:
                            self.Z[i] = self.img[int(self.pts[i,1]),int(self.pts[i,0])]
                        self.Z_ext[i] = self.img[int(self.pts[i, 1]), int(self.pts[i, 0])]
                    homogen3dal1[i,0] = int(self.pts[i,1])
                    homogen3dal1[i, 1] = int(self.pts[i, 0])
                    homogen3dal1[i, 2] = self.Z_ext[i]
                    homogen3dal1[i, 3] = 1
                t1 = time.time()
                if verbose:
                    print("Camera Computation time: ", t1 - t0)
                #compute triangluation and frontalize
                t0 = time.time()
                tri = self.generate_3Dtriangulation_mapping_data(self.ptsxv,self.img)
                t1 = time.time()
                if verbose:
                    print("Triangulation time: ", t1 - t0)
                #TODO is X and img the same thing?
                t0 = time.time()
                mapData = self.do_3Dtexture_mapping_with_delaunay(landmarks,faceImg,self.img,tri,P)
                t1 = time.time()
                if verbose:
                    print("Mapping time: ", t1 - t0)
                mapped_img = mapData[0]
                dgenmap = mapData[1]
                t0 = time.time()
                mapped_img = mapped_img[mapped_img.shape[0]-mapped_img.shape[1]:mapped_img.shape[0]]
                self.modelLandmarks = self.modelLandmarks-[0,50]
                mapped_img_f = np.fliplr(mapped_img)
                frontal_raw = mapped_img
                if side == 1:
                    mapped_img = mapped_img_f
                    mapped_img_f = frontal_raw
                frontal_sym1 = np.hstack((mapped_img[:,0:73],mapped_img_f[:,77:mapped_img_f.shape[1]]))
                frontal_sym2 = np.hstack((mapped_img[:, 0:74], mapped_img_f[:, 76: mapped_img_f.shape[1]]))
                frontal_sym3 = np.hstack((mapped_img[:, 0:75], mapped_img_f[:, 75: mapped_img_f.shape[1]]))
                frontal_sym4 = np.hstack((mapped_img[:, 0:76], mapped_img_f[:, 74: mapped_img_f.shape[1]]))
                frontal_sym5 = np.hstack((mapped_img[:, 0:77], mapped_img_f[:, 73: mapped_img_f.shape[1]]))
                images = [frontal_sym1,frontal_sym2,frontal_sym3,frontal_sym4,frontal_sym5]
                bestsymIndex = self.bestSymIndex((frontal_sym1,frontal_sym2,frontal_sym3,frontal_sym4,frontal_sym5))
                t1 = time.time()
                if verbose:
                    print("Face flip time: ", t1 - t0)
                t0 = time.time()
                bestImage = images[bestsymIndex]
                t1 = time.time()
                if verbose:
                    print("Best image time: ", t1 - t0)
                if side == 0 or True:
                    bestImage = frontal_raw
        return bestImage,dgenmap


    def bestSymIndex(self,images):
        minDeriv = sys.float_info.max
        index = -1
        minDerivimg = []
        i = 0
        for face in images:
            backtorgb = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            centerface = face[:,int(max(face.shape[1]/2.0-face.shape[1]/10.0,0)):int(min(face.shape[1]/2.0+face.shape[1]/10.0,face.shape[1]))]
            diriv = cv2.Laplacian(centerface,cv2.CV_16S,3,1)
            laplac = cv2.convertScaleAbs(diriv)
            total = np.abs(laplac).sum()*(1.0+i)/10.0
            if total*1.3 < minDeriv:
                minDeriv = total
                index = i
                minDerivimg = laplac
            i+=1
        return index
    def simplePoseDetector(self,landmarks,img,chull=None):
        if chull is None:
            chull = cv2.convexhull(landmarks)
        x,y,w,h = cv2.boundingRect(chull)
        centerpoints = np.asarray([27,28,29,30,31,32,33,34,35])
        centroidx = landmarks[centerpoints,0].mean()
        centroidy = landmarks[centerpoints, 1].mean()
        faceCentroidx = chull[:,0,0].mean()
        faceCentroidy = chull[:,0,1].mean()
        if abs(centroidx-faceCentroidx) <= w*.15:
            return 0
        if centroidx-faceCentroidx > 0:
            return -1
        return 1
    def estimateCamera(self,landmarks):
        Aout,Rout,Tout = self.runCalibration(self.model3d.width,self.model3d.height,landmarks.shape[0],landmarks,self.model3d.threedee,self.model3d.outA,0,0,1,0,None,None)
        return (Aout,Rout,Tout)
        pass
    def runCalibration(self,width,height,numPoints,imgPointsIn,objPointsIn,cameraMatrix,max_mse,usePosit,onlyExtrinsic,useExtrinsicGuess, RotationMatrix, TranslationMatrix):
        dist_coeffs = None
        RotationMatrix = np.asarray(RotationMatrix,dtype='float32')
        TranslationMatrix = np.asarray(TranslationMatrix,dtype='float32')
        cameraMatrix = np.asarray(cameraMatrix,dtype='float32')
        if onlyExtrinsic:
            camera_matrix = cameraMatrix
            # rotation_matrices_tmp, _ = cv2.Rodrigues(RotationMatrix)
            retval, rvec, tvec = cv2.solvePnP(objPointsIn, imgPointsIn, cameraMatrix, dist_coeffs)
            rotation_matrices,_ = cv2.Rodrigues(rvec)
            # rotation_matrices = RotationMatrix
            translation_vectors = tvec
        else:
            retval, camera_matrix, dist_coeffs, rotation_matrices, translation_vectors = cv2.calibrateCamera(objPointsIn, imgPointsIn, (height,width), None, None)
        AOutput = cameraMatrix.transpose()
        ROutput = RotationMatrix.transpose()
        TOutput = TranslationMatrix.transpose()
        return (AOutput,ROutput,TOutput)
        # image_points,jacobian = cv2.projectPoints(objPointsIn,rotation_matrices,translation_vectors,camera_matrix,dist_coeffs)

    def generateDefaultLandmarkTemplate(self,scale=1):
        pts = np.zeros((68,2),dtype='float32')
        pts[36, 0] = 28.1718;
        pts[36, 1] = 79.1260;
        pts[39, 0] = 58.0954;
        pts[39, 1] = 78.5153;
        pts[37, 0] = 37.6374;
        pts[37, 1] = 74.5458;
        pts[38, 0] = 47.1031;
        pts[38, 1] = 73.0191;
        pts[40, 0] = 48.9351;
        pts[40, 1] = 84.6221;
        pts[41, 0] = 37.6374;
        pts[41, 1] = 84.6221;

        pts[42, 0] = 95.3473;
        pts[42, 1] = 77.5992;
        pts[43, 0] = 105.1183;
        pts[43, 1] = 73.9351;
        pts[44, 0] = 114.2786;
        pts[44, 1] = 74.2405;
        pts[45, 0] = 123.4389;
        pts[45, 1] = 79.7366;
        pts[46, 0] = 114.2786;
        pts[46, 1] = 82.7901;
        pts[47, 0] = 104.2023;
        pts[47, 1] = 82.7901;

        pts[17, 0] = 19.9275;
        pts[17, 1] = 68.4389;
        pts[18, 0] = 28.4771;
        pts[18, 1] = 61.4160;
        pts[19, 0] = 37.3321;
        pts[19, 1] = 59.2786;
        pts[20, 0] = 46.7977;
        pts[20, 1] = 61.7214;
        pts[21, 0] = 58.0954;
        pts[21, 1] = 65.3855;

        pts[22, 0] = 95.6527;
        pts[22, 1] = 66.9122;
        pts[23, 0] = 103.5916;
        pts[23, 1] = 61.4160;
        pts[24, 0] = 113.0573;
        pts[24, 1] = 58.3626;
        pts[25, 0] = 124.0496;
        pts[25, 1] = 60.8053;
        pts[26, 0] = 131.9885;
        pts[26, 1] = 66.3015;

        pts[27, 0] = 77.0267;
        pts[27, 1] = 73.0191;
        pts[28, 0] = 76.7214;
        pts[28, 1] = 88.8969;
        pts[29, 0] = 77.0267;
        pts[29, 1] = 102.6374;
        pts[30, 0] = 77.3321;
        pts[30, 1] = 117.2939;

        pts[31, 0] = 60.8435;
        pts[31, 1] = 125.8435;
        pts[32, 0] = 68.1718;
        pts[32, 1] = 127.6756;
        pts[33, 0] = 76.7214;
        pts[33, 1] = 129.2023;
        pts[34, 0] = 84.9656;
        pts[34, 1] = 126.7595;
        pts[35, 0] = 93.2099;
        pts[35, 1] = 125.2328;

        pts[48, 0] = 56.5687;
        pts[48, 1] = 153.0191;
        pts[49, 0] = 63.8969;
        pts[49, 1] = 148.1336;
        pts[50, 0] = 71.2252;
        pts[50, 1] = 145.9962;
        pts[51, 0] = 76.4160;
        pts[51, 1] = 148.7443;
        pts[52, 0] = 81.9122;
        pts[52, 1] = 145.6908;
        pts[53, 0] = 92.5992;
        pts[53, 1] = 148.7443;
        pts[54, 0] = 101.7595;
        pts[54, 1] = 155.4618;

        pts[55, 0] = 94.1260;
        pts[55, 1] = 159.1260;
        pts[56, 0] = 85.2710;
        pts[56, 1] = 164.0115;
        pts[57, 0] = 76.1107;
        pts[57, 1] = 164.3168;
        pts[58, 0] = 66.0344;
        pts[58, 1] = 163.0954;
        pts[59, 0] = 57.7901;
        pts[59, 1] = 159.1260;

        pts[60, 0] = 60.8435;
        pts[60, 1] = 153.0191;
        pts[61, 0] = 68.4771;
        pts[61, 1] = 152.4084;
        pts[62, 0] = 74.5840;
        pts[62, 1] = 153.6298;
        pts[63, 0] = 81.6069;
        pts[63, 1] = 152.4084;
        pts[64, 0] = 90.1565;
        pts[64, 1] = 153.9351;
        pts[65, 0] = 82.8282;
        pts[65, 1] = 157.2939;
        pts[66, 0] = 74.8893;
        pts[66, 1] = 156.9885;
        pts[67, 0] = 68.7824;
        pts[67, 1] = 155.4618;

        pts[0, 0] = 6.1870;
        pts[0, 1] = 83.7061;
        pts[1, 0] = 8.0191;
        pts[1, 1] = 103.8588;
        pts[2, 0] = 11.3779;
        pts[2, 1] = 122.4847;
        pts[3, 0] = 16.5687;
        pts[3, 1] = 140.5000;
        pts[4, 0] = 24.5076;
        pts[4, 1] = 156.9885;
        pts[5, 0] = 36.4160;
        pts[5, 1] = 171.9504;
        pts[6, 0] = 47.1031;
        pts[6, 1] = 184.1641;
        pts[7, 0] = 61.7595;
        pts[7, 1] = 193.3244;
        pts[8, 0] = 77.0267;
        pts[8, 1] = 198.2099;
        pts[9, 0] = 93.6527;
        pts[9, 1] = 194.5458;
        pts[10, 0] = 111.8359;
        pts[10, 1] = 184.1641;
        pts[11, 0] = 121.8282;
        pts[11, 1] = 172.8664;
        pts[12, 0] = 131.6832;
        pts[12, 1] = 159.1260;
        pts[13, 0] = 138.4008;
        pts[13, 1] = 142.0267;
        pts[14, 0] = 141.7595;
        pts[14, 1] = 120.9580;
        pts[15, 0] = 145.4237;
        pts[15, 1] = 100.1947;
        pts[16, 0] = 144.8130;
        pts[16, 1] = 80.3473;
        #We mirror the 3D model and coordinates to make them symmetric
        eye_mirror_perms = np.asarray([17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 28, 29, 30,
         31, 36, 35, 34, 33, 32, 46, 45, 44, 43, 48, 47, 40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58,
         57, 56, 65, 64, 63, 62, 61, 68, 67, 66],dtype=np.int32)
        pts = pts*scale
        pts2 = np.zeros((pts.shape),dtype='float32')
        pts2[:,0] = (150*scale)-pts[eye_mirror_perms-1,0]
        pts2[:,1] = pts[eye_mirror_perms-1,1]
        pts = np.asarray((pts+pts2)/2,dtype=np.int32)

        #We add a few additional coordiantes to make the facial area bigger (not really needed)
        self.pts = pts
        self.ptsx = pts
        self.ptsxv = self.ptsx
        self.modelLandmarks = self.ptsxv
        pts1 = np.int32(np.asarray([[22,14],[75,1],[128,14]],dtype='float32')*scale)
        self.pts = np.vstack((pts,pts1))
        maskHull = cv2.convexHull(self.pts)
        self.mask3D = np.int32(cv2.fillPoly(np.zeros((self.img.shape)), np.int32([maskHull]), 1))-1
        #Produce the Z values of the annotated points
        self.Z = np.zeros((self.ptsx.shape[0],1),dtype='int32')
        self.Z_ext = np.zeros((self.pts.shape[0],1),dtype='int32')
        self.Z[:,0] = self.img[self.pts[:68,1],self.pts[:68,0]]
        self.Z_ext = self.img[self.pts[:,1],self.pts[:,0]]
        #Assemble 3D points into homogenous form
        x3d = np.asarray(np.append(pts[:68],self.Z,axis=1),dtype='float32')
        self.homogen3D = np.hstack((x3d,np.ones((x3d.shape[0],1),dtype='float32')))
        mua = cv2.reduce(x3d,0,cv2.REDUCE_AVG)
        muaRep = cv2.repeat(mua,x3d.shape[0],1)
        x3d_zm = x3d - muaRep
        powx3d_zm = np.square(x3d_zm)
        d_sq = cv2.reduce(powx3d_zm[:,0],1,cv2.REDUCE_SUM)
        mean_sq = d_sq.mean()
        s= math.sqrt(mean_sq/3)
        scalexyz = math.sqrt(powx3d_zm.sum()/(3*68))
        scale = np.ones((1,3),dtype='float32')/scalexyz
        self.U = np.eye(3,dtype='float32')/scalexyz
        self.U = np.vstack((self.U,np.zeros((1,3),dtype='float32')))
        mscaletmp = -(scale*mua).transpose()
        mscaletmp = np.vstack((mscaletmp,np.ones((1,1))))
        self.U = np.hstack((self.U,mscaletmp))
        self.homo_x3d = np.hstack((x3d,np.ones((68,1),dtype='float32')))
        self.normX3d = np.dot(self.U,self.homo_x3d.transpose())
    def get_tri_data(self, tri, ind, img):
        triang = Triang()
        triang.target = np.asarray(tri.trianglePoints[ind][0:3])
        triang.targetZ = np.zeros((3,4),dtype='float32')
        triang.xmin = triang.target[:,0].min()
        triang.ymin = triang.target[:,1].min()
        triang.xmax = triang.target[:,0].max()
        triang.ymax = triang.target[:,1].max()
        ref = tri.listOfIndexesPerTriangle[ind]
        triang.ref = np.asarray(ref)
        triang.coor = np.zeros((triang.ref.shape[0],4),dtype='float32')
        Z = np.zeros((triang.ref.shape[0],1),dtype='float32')
        Z[:,0] = img[triang.ref[:,1],triang.ref[:,0]]
        triang.coor[:,0:2] = triang.ref
        triang.coor[:,2] = img[triang.ref[:,1],triang.ref[:,0]]
        triang.coor[:,3] = 1+    np.zeros((triang.ref.shape[0]),dtype='float32')
        return triang

    def draw_delaunay(self,img, subdiv, delaunay_color):

        triangleList = subdiv.getTriangleList();
        size = img.shape
        r = (0, 0, size[1], size[0])
        count = 0
        for t in triangleList:

            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
                cv2.line(img, pt1, pt2, delaunay_color, 1)
                cv2.line(img, pt2, pt3, delaunay_color, 1)
                cv2.line(img, pt3, pt1, delaunay_color, 1)
            count+=1
    def generate_3Dtriangulation_mapping_data(self,shape_points,img):#shape_points must be a list of tuple points
        tri = Tri()
        imgsize_y = img.shape[0]
        imgsize_x = img.shape[1]
        tri.size = img.shape
        trimax1 = np.max(shape_points[:,0])
        trimax2 = np.max(shape_points[:,1])
        trimin1 = np.max(shape_points[:,0])
        trimin2 = np.min(shape_points[:,1])
        tri.mins = (trimin1,trimin2)
        tri.maxs = (trimax1,trimax2)
        #do triangulation
        #normalize coordinates
        tri.points = shape_points
        trirect = (0,0,img.shape[1],img.shape[0])
        subdiv = cv2.Subdiv2D(trirect)
        # points_list = []
        # for i in range(shape_points.shape[0]):
        #     points_list.append((shape_points[i,0], shape_points[i,1]))
        tupleshapes = []
        for r in range(shape_points.shape[0]):
            tupleshapes.append((int(shape_points[r,0]),int(shape_points[r,1])))
        subdiv.insert(tupleshapes)
        triangleList = subdiv.getTriangleList()
        # self.draw_delaunay(img,subdiv,(255,0,0))

        #get triangle count
        tri.count = len(triangleList)
        centroids = []
        pos1 = None
        pos2 = None
        pos3 = None

        for i in range(triangleList.shape[0]):
            t = triangleList[i]
            pt = ((t[0],t[1]),(t[2],t[3]),(t[4],t[5]))
            try:
                pos1 = tupleshapes.index(pt[0])
                pos2 = tupleshapes.index(pt[1])
                pos3 = tupleshapes.index(pt[2])
                if pos1 is not None and pos2 is not None and pos3 is not None:
                    tri.trianglePoints.append(pt)
                    tri.indices.append((pos1,pos2,pos3))
                    centroids.append((float(pt[0][0]+pt[1][0]+pt[2][0])/3,float(pt[0][1]+pt[1][1]+pt[2][1])/3))

            except:
                pass
        tri.create_tri_map(self.mask3D)
        #create coordinates of target image
        tri.count = len(tri.trianglePoints)
        tri.listOfIndexesPerTriangle = {}
        tri.inds = np.zeros(img.shape,dtype='float32')
        vis = img
        c = 0
        maxInd = 0
        minInd = 10000000
        tlength = tri.inds.shape[1] * tri.inds.shape[0]
        # bar = progressbar.ProgressBar(max_value=tlength)
        c = 0
        for x in range(tri.inds.shape[1]):
            for y in range(tri.inds.shape[0]):
                ind = tri.findTriangleIndex((x,y))
                # print(ind)
                if ind >= 0 and ind < len(tri.trianglePoints):
                    if ind in tri.listOfIndexesPerTriangle:
                        tri.listOfIndexesPerTriangle[ind].append((x,y))
                    else:
                        tri.listOfIndexesPerTriangle[ind] = [(x,y)]
                    if ind < minInd: minInd = ind
                    if ind > maxInd: maxInd = ind
                tri.inds[y,x] = ind
                c+=1
                # bar.update(c)
        #construct triangle data
        for i in range(tri.count):
            tri.triangles.append(self.get_tri_data(tri,i,img))
        return tri
    def do_3Dtexture_mapping_with_delaunay(self,probe_mesh_pts,probe_image,img,tri,P):
        #clean probe image - remove all parts of image except for the face covered by the landmarks
        cx = 1
        if len(probe_image.shape) > 2:
            cx = probe_image.shape[2]
        mm = probe_image.shape[0]
        nn = probe_image.shape[1]
        dd = None
        if cx > 1:
            dd = probe_image.shape[2]
        #normalize probe shape coordinates and find triangle points
        probe_pts = probe_mesh_pts
        #initialize output
        mapped_img = np.zeros((tri.size[0],tri.size[1],cx),dtype=np.uint8)
        #do the mapping
        points1 = probe_mesh_pts
        cont = 1
        vis2 = np.copy(probe_image)
        vis1 = np.copy(img)
        minv = np.min(vis1)
        maxv = np.max(vis1)
        vis1 /= maxv
        vis3 = np.copy(mapped_img)
        pixelLocs = np.arange(probe_image.shape[0]*probe_image.shape[1]).reshape(probe_image.shape[:2])
        mappedPixelLocs = np.zeros(tri.size,dtype=np.int32)
        degenerateMap = np.zeros(tri.size,dtype=np.int32)
        for i in range(tri.count):
            coordinates = tri.triangles[i].ref
            Z = np.int32(np.dot(P,(tri.triangles[i].coor.transpose())))
            probe_x = Z[0]
            probe_y = Z[1]
            mappedPixelLocs[coordinates[:, 1], coordinates[:, 0]] = pixelLocs[probe_y, probe_x]
            mapped_img[coordinates[:, 1], coordinates[:, 0]] = probe_image[probe_y, probe_x]

            # fig = plt.figure()
            # plt.ion()
            # for j in np.arange(0,coordinates.shape[0],6):
            #     vis1 = cv2.circle(vis1,tuple(coordinates[j]),1,(1,1,1),-1)
            #     vis2 = cv2.circle(vis2,(int(probe_x[j]),int(probe_y[j])),1,(255,0,0),-1)
            #     radius =6
            #     # vis3[coordinates[j][1],coordinates[j][0]] = mapped_i[coordinates[j][1],coordinates[j][0]]
            #     for x in range(radius):
            #         for y in range(radius):
            #             if x >= 0 and y >=0 and x < vis3.shape[1] and y < vis3.shape[0]:
            #                 vis3[int((coordinates[j][1]-radius/2)+y),int((coordinates[j][0]-radius/2)+x)] = mapped_img[int((coordinates[j][1]-radius/2)+y),int((coordinates[j][0]-radius/2)+x)]
            #     plt.close()
            #     f, axarr = plt.subplots(1, 3)
            #     axarr[0].imshow(vis1)
            #     axarr[1].imshow(vis2)
            #     axarr[2 ].imshow(vis3)
            #     plt.draw()
            #     if j == 0 and i == 0:
            #         plt.waitforbuttonpress()
            #     else:
            #         plt.waitforbuttonpress(timeout=.1)
            #     # plt.waitforbuttonpress()
            # uniqueMappedCoords = np.array(list(set(tuple(p) for p in coordinates)))
            # uniqueProbePoints = np.array(list(set(tuple(p) for p in Z[0:2,:].transpose())))



        final_mapped_img = mapped_img
        # shared_pixelLocs = mp.Array(ctypes.c_int32, mappedPixelLocs)
        # shared_dmap = mp.Array(ctypes.c_int32,degenerateMap)


        windowSize = 15
        step = 6
        degenerateMap = self.degenerateBuilder.buildDegenerateMap(mappedPixelLocs,final_mapped_img)
        # locationPatches = feature_extraction.image.extract_patches_2d(mappedPixelLocs, (windowSize,windowSize))


        # pool = Pool()
        # output = pool.map(countUnique, (patch for patch in locationPatches))
        # pad = pad = int((windowSize-1)/2)
        # pool.appl
        # patchOutput = np.asarray(output,dtype=np.int32).reshape((degenerateMap.shape[0]-(windowSize-1),degenerateMap.shape[1]-(windowSize-1)))
        # degenerateMap[pad:patchOutput.shape[0] + pad, pad:patchOutput.shape[1] + pad] = patchOutput
        # for x in range(0,mappedPixelLocs.shape[1]-math.ceil(windowSize),step):
        #     for y in range(0,mappedPixelLocs.shape[0]-math.ceil(windowSize),step):
        #         centerx = x+int(windowSize/2)
        #         centery = y+int(windowSize/2)
        #         if np.sum(final_mapped_img[centery,centerx]) > 0:
        #             window = mappedPixelLocs[y:y+windowSize,x:x+windowSize]
        #             nUnique = len(np.unique(window))
        #             degenerateMap[centery:centery+step,centerx:centerx+step] = nUnique
        degenerateMap = np.uint8((degenerateMap/degenerateMap.max())*255)
        mask3D = np.uint8(cv2.resize(self.mask3D,(degenerateMap.shape[1],degenerateMap.shape[0]))+1)
        kernel = np.ones((20, 20), np.uint8)
        mask3D = cv2.erode(mask3D, kernel, iterations=1)
        ret,replace_mask = cv2.threshold(degenerateMap,100,255,cv2.THRESH_BINARY_INV)
        replace_mask *= mask3D
        # inpainted = self.inPainter.inpaint(final_mapped_img,replace_mask/255)
        # inpainted = cv2.inpaint(final_mapped_img, replace_mask, 3, cv2.INPAINT_NS)
        r = np.uint8(replace_mask/255)
        r = np.dstack((r,r,r))
        srcImg = np.fliplr(final_mapped_img)
        #inpainted = poissonblending.blend(final_mapped_img,srcImg,r[:,:,0])
        #inpainted = mirrorInpaint(final_mapped_img, r)
        inpainted = []
        face_masked = final_mapped_img*(1-r)
        return (face_masked,inpainted)

    def face_orientation(self,frame, landmarks):
        size = frame.shape  # (height, width, color_channel)

        image_points = np.array([
            (landmarks[0][0], landmarks[0][1]),  # Nose tip
            (landmarks[51][0], landmarks[51][1]),  # Chin
            (landmarks[11][0], landmarks[11][1]),  # Left eye left corner
            (landmarks[22][0], landmarks[22][1]),  # Right eye right corne
            (landmarks[34][0], landmarks[34][1]),  # Left Mouth corner
            (landmarks[40][0], landmarks[40][1])  # Right mouth corner
        ], dtype=np.float32)

        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-165.0, 170.0, -135.0),  # Left eye left corner
            (165.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ],dtype=np.float32)

        # Camera internals

        center = (size[1] / 2, size[0] / 2)
        focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float32
        )

        dist_coeffs = np.zeros((4, 1),dtype=np.float32)  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        axis = np.float32([[500, 0, 0],
                           [0, 500, 0],
                           [0, 0, 500]])

        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix,
                                           dist_coeffs)
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (landmarks[0][0], landmarks[0][1])

    def genPoseOutputImage(self,frame, landmarks):

        imgpts, modelpts, rotate_degree, nose = self.face_orientation(frame, landmarks)

        cv2.line(frame, nose, tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # GREEN
        cv2.line(frame, nose, tuple(imgpts[0].ravel()), (255, 0,), 3)  # BLUE
        cv2.line(frame, nose, tuple(imgpts[2].ravel()), (0, 0, 255), 3)  # RED

        # remapping = [2, 3, 0, 4, 5, 1]
        # for index in range(int(len(landmarks) / 2)):
        #     random_color = tuple(np.random.random_integers(0, 255, size=3))
        #
        #     cv2.circle(frame, (landmarks[index * 2], landmarks[index * 2 + 1]), 5, random_color, -1)
        #     cv2.circle(frame, tuple(modelpts[remapping[index]].ravel().astype(int)), 2, random_color, -1)
        #
        #
        #     #    cv2.putText(frame, rotate_degree[0]+' '+rotate_degree[1]+' '+rotate_degree[2], (10, 30),
        #     #                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
        #     #                thickness=2, lineType=2)

        for j in range(len(rotate_degree)):
            cv2.putText(frame, ('{:05.2f}').format(float(rotate_degree[j])), (10, 30 + (50 * j)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

        return frame

if __name__ == "__main__":
    imgdir = sys.argv[1]
    outputDir = sys.argv[2]
    numcores = int(sys.argv[3])
    runImageDir(imgdir,outputDir,numcores)
    #testFrontalize()
#runImageDir('/Users/joel/Documents/Projects/Thesis/frontalization_output/flynn','/Users/joel/Documents/Projects/Thesis/frontalization_output/output')