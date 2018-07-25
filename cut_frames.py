import cv2 as cv
import os

videos = os.listdir(r"C:\Users\t-danur\Downloads\videos")
video_count = 1
for video in videos:
  cap = cv.VideoCapture("C:\\Users\\t-danur\Downloads\\videos\\" + video)
  success, frame = cap.read()
  count = 0
  while success:
    if count%300 ==0:
      cv.imwrite("C:\\Users\\t-danur\Downloads\\frames\\frame%d_%d.jpg" % (video_count ,count) , frame)
      # Display the resulting frame
      cv.imshow('Frame', frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
      break
    count += 1
    success, frame = cap.read()
  video_count = 2

  # When everything done, release the video capture object
  cap.release()

  # Closes all the frames
  cv.destroyAllWindows()