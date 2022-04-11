import cv2


image_size = (1280, 720) # Provides a good balance between performance and FPS. 480p: 0.0465s, 720p: 0.145s
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print('failed to grab frame')
        break
    frame = cv2.resize(frame, image_size)
    cv2.imshow('Output', frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        break
    elif k%256 == 32:
        img_name = 'calibration_image_{}.png'.format(img_counter)
        cv2.imwrite(img_name, frame)
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
