# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2


from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
# vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def detect_motion(frameCount, width = 800):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock

    path_angry = "angry/"
    path_happy = "happy/"
    path_sad = "sad/"
    path_neutral = "neutral/"

    angry_latest = 100000+1
    happy_latest = 100000 + 1
    sad_latest = 100000 + 1
    neutral_latest = 100000 + 1

    # initialize the motion detector and the total number of frames
    # read thus far
    # md = SingleMotionDetector(accumWeight=0.1)

    # parameters for loading data and images
    detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
    emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

    # hyper-parameters for bounding boxes shape
    # loading models
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
                "neutral"]
    total = 0
    label = 'neutral'

    # loop over frames from the video stream

    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        frame = imutils.resize(frame, width=width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        if total % frameCount == 0:
            faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)

            if len(faces) > 0:
                faces = sorted(faces, reverse=True,
                               key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = faces
                # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]


                cv2.putText(frame, label, (fX, fY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                # cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
                #               (0, 0, 255), 2)
            else:
                continue



        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        # if total > frameCount:
        #     # detect motion in the image
        #     motion = md.detect(gray)
        #
        #     # cehck to see if motion was found in the frame
        #     if motion is not None:
        #         # unpack the tuple and draw the box surrounding the
        #         # "motion area" on the output frame
        #         (thresh, (minX, minY, maxX, maxY)) = motion
        #         cv2.rectangle(frame, (minX, minY), (maxX, maxY),
        #                       (0, 0, 255), 2)

        # update the background model and increment the total number
        # of frames read thus far
        # md.update(gray)
        total += 1

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            if label == "angry":
                outputFrame = cv2.imread(path_angry+"frame-"+str(angry_latest)[1:]+".jpg")
                outputFrame = imutils.resize(outputFrame, width=width)
                angry_latest+=1
            elif label == "neutral":
                outputFrame = cv2.imread(path_angry + "frame-" + str(neutral_latest)[1:] + ".jpg")
                outputFrame = cv2.imread("neutral.png")
                outputFrame = imutils.resize(outputFrame, width=width)
                neutral_latest+=1
            elif label == "happy":
                outputFrame = cv2.imread(path_angry + "frame-" + str(happy_latest)[1:] + ".jpg")
                outputFrame = cv2.imread("happy.jpeg")
                outputFrame = imutils.resize(outputFrame, width=width)
                happy_latest+=1
            elif label == "sad":
                outputFrame = cv2.imread(path_angry + "frame-" + str(sad_latest)[1:] + ".jpg")
                outputFrame = cv2.imread("sad.jpeg")
                outputFrame = imutils.resize(outputFrame, width=width)
                sad_latest+=1


            elif label == "scared":
                outputFrame = cv2.imread("scared.png")
                outputFrame = imutils.resize(outputFrame, width=width)

            elif label == "surprised":
                outputFrame = cv2.imread("surprised.jpg")
                outputFrame = imutils.resize(outputFrame, width=width)

            elif label == "disgust":
                outputFrame = cv2.imread("disgust.jpg")
                outputFrame = imutils.resize(outputFrame, width=width)

            # outputFrame = frame.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=100,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()