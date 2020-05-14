import cv2
import dlib

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

eyes_position = []

while (True):
    ret, frame = cap.read()  # read frame by frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray frame
    faces = detector(gray)  # detect face
    for face in faces:

        landmarks = predictor(gray, face)  # create face landmarks

        # coords for eyes
        left_coords_right_eye = landmarks.part(36).x - 10, landmarks.part(37).y - 10
        right_coords_right_eye = landmarks.part(39).x + 10, landmarks.part(41).y + 10

        left_coords_left_eye = landmarks.part(42).x - 10, landmarks.part(43).y - 10
        right_coords_left_eye = landmarks.part(45).x + 10, landmarks.part(47).y + 10

        # make right and left eye roi
        left_eye_roi = frame[left_coords_right_eye[1]: right_coords_right_eye[1],
                       left_coords_right_eye[0]: right_coords_right_eye[0]]
        right_eye_roi = frame[left_coords_right_eye[1]: right_coords_right_eye[1],
                        left_coords_right_eye[0]: right_coords_right_eye[0]]

        # make rois gray
        gray_right_roi = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)
        gray_left_roi = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)

        # blur
        gray_right_roi = cv2.GaussianBlur(gray_right_roi, (3, 3), 0)
        gray_left_roi = cv2.GaussianBlur(gray_left_roi, (3, 3), 0)

        # thresholds
        _, threshold_right = cv2.threshold(gray_right_roi, 60, 255, cv2.THRESH_BINARY_INV)
        _, threshold_left = cv2.threshold(gray_left_roi, 60, 255, cv2.THRESH_BINARY_INV)

        # grab countours
        contours_left, hierarchy = cv2.findContours(threshold_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_right, hierarchy = cv2.findContours(threshold_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # for the largest area calculate rect size and mid point
        contours = sorted(contours_left, key=lambda x: cv2.contourArea(x), reverse=True)
        for i, cnt in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(cnt)
            mid_point = (x + w / 2, y + h / 2)
            print(mid_point)  # print eye position (roi sizes)
            eyes_position.append(mid_point)
            x = x + left_coords_left_eye[0]
            y = y + left_coords_left_eye[1]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # draw rect o eye
            break

        # for the largest area calculate rect size and mid point
        contours = sorted(contours_right, key=lambda x: cv2.contourArea(x), reverse=True)
        for i, cnt in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(cnt)
            mid_point = (x + w / 2, y + h / 2)
            print(mid_point)  # print eye position (roi sizes)
            eyes_position.append(mid_point)
            x = x + left_coords_right_eye[0]
            y = y + left_coords_right_eye[1]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # draw rect o eye
            break

        cv2.imshow("frame", frame)  # show frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # q to exit
        break

# quit
cap.release()
cv2.destroyAllWindows()
