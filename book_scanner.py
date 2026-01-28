import cv2
import pytesseract
import numpy as np

def preprocess_for_ocr(image, mode='balanced'):
    """
    Preprocessing modes:
    - minimal: just grayscale + resize
    - balanced: grayscale + resize + CLAHE + binarization
    - tophat: grayscale + resize + top-hat transform 
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to 2x for better OCR (Tesseract likes ~300 DPI)
    height, width = gray.shape
    gray = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    
    if mode == 'minimal':
        return gray
    
    # CLAHE + Binarization
    if mode == 'balanced':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    # Top-hat transform
    if mode == 'tophat':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        combined = cv2.add(gray, tophat)
        combined = cv2.subtract(combined, blackhat)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(combined)
        
        return enhanced
    
    return gray

print("Opening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera ready!")
print("Press 's' to scan | 'q' to quit")
print("Press '1' = minimal | '2' = balanced (CLAHE + binary) | '3' = top-hat")

# Default mode
process_mode = 'minimal'
mode_names = {'minimal': '1-Minimal', 'balanced': '2-Balanced+Binary', 'tophat': '3-TopHat'}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.putText(frame, mode_names[process_mode], (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Book Scanner', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('1'):
        process_mode = 'minimal'
        print("Mode: Minimal (grayscale + resize)")
    elif key == ord('2'):
        process_mode = 'balanced'
        print("Mode: Balanced (+ CLAHE + binarization)")
    elif key == ord('3'):
        process_mode = 'tophat'
        print("Mode: Top-hat (for colored/textured backgrounds)")
    
    elif key == ord('s'):
        print(f"\n=== Scanning with {process_mode} mode ===")
        
        # Preprocess
        processed = preprocess_for_ocr(frame, mode=process_mode)
        
        # Show processed image
        display_processed = cv2.resize(processed, (640, 480))
        cv2.imshow('Processed', display_processed)
        
        # OCR on the full-size processed image
        custom_config = r'--oem 3 --psm 11'
        
        data = pytesseract.image_to_data(processed, config=custom_config, 
                                         output_type=pytesseract.Output.DICT)
        
        # Draw boxes and text 
        n_boxes = len(data['text'])
        found_text = False
        result_frame = frame.copy()
        
        all_text = []
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if conf > 40 and len(text) > 1:
                found_text = True
                # Coordinates are 2x because we resized, so divide by 2
                (x, y, w, h) = (data['left'][i]//2, data['top'][i]//2, 
                               data['width'][i]//2, data['height'][i]//2)
                
                # Draw box
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw text
                cv2.putText(result_frame, text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                print(f"'{text}' | Confidence: {conf}%")
                all_text.append(text)
        
        if found_text:
            print(f"\nFull text: {' '.join(all_text)}")
        else:
            print("No text detected - try different mode")
        
        cv2.imshow('Results', result_frame)
        cv2.waitKey(3000)
        cv2.destroyWindow('Processed')
        cv2.destroyWindow('Results')
        print("=== Done ===\n")
    
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
