import cv2
import numpy as np
import tensorflow as tf

model_path = 'best_model.keras'
model = tf.keras.models.load_model(model_path)

img_size = 224
closed_count = 0
closed_threshold = 10 
drowsy = False

code = {'Closed':0, 'Open':1, 'no_yawn':2, 'yawn':3}
rev_code = {v: k for k, v in code.items()}

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img = cv2.resize(frame, (img_size, img_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_batch = np.expand_dims(img_rgb, axis=0)

    processed_iter = datagen.flow(img_batch, batch_size=1, shuffle=False)
    processed_img = next(processed_iter)

    pred = model.predict(processed_img)
    pred_class = np.argmax(pred)
    label = rev_code[pred_class]

    if label == 'Closed':
        closed_count += 1
    else:
        closed_count = 0  # Reset if eyes are open or yawning stops

    if label == 'yawn' or closed_count >= closed_threshold:
        drowsy = True
    else:
        drowsy = False
    
    status_text = "DROWSY!" if drowsy else "Awake"
    color = (0, 0, 255) if drowsy else (0, 255, 0)

    cv2.putText(frame, f'Prediction: {label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f'Status: {status_text}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Drowsiness Detection', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()