import cv2
import os
from google.cloud import vision
import requests
import json

# Check credentials 
if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
    print("ERROR: GOOGLE_APPLICATION_CREDENTIALS not set!")
    print("Run: export GOOGLE_APPLICATION_CREDENTIALS='/home/pi/vision-credentials.json'")
    exit(1)

print("Initializing Vision API...")
try:
    vision_client = vision.ImageAnnotatorClient()
    print("Vision API initialized successfully!")
except Exception as e:
    print(f"Failed to initialize Vision API: {e}")
    exit(1)

def lookup_book_openlibrary(title_hint=None, author_hint=None):
    """
    Query Open Library API for book metadata
    """
    print(f"\nLooking up book: title='{title_hint}', author='{author_hint}'")
    
    url = "https://openlibrary.org/search.json?"
    params = []
    
    if title_hint:
        params.append(f"title={title_hint}")
    if author_hint:
        params.append(f"author={author_hint}")
    
    if not params:
        return None
    
    full_url = url + "&".join(params)
    
    try:
        response = requests.get(full_url, timeout=5)
        data = response.json()
        
        if data.get('docs') and len(data['docs']) > 0:
            book = data['docs'][0]
            
            result = {
                'title': book.get('title', 'Unknown'),
                'author_full': book.get('author_name', ['Unknown'])[0] if book.get('author_name') else 'Unknown',
                'first_publish_year': book.get('first_publish_year', 'Unknown'),
                'isbn': book.get('isbn', ['N/A'])[0] if book.get('isbn') else 'N/A'
            }
            
            print(f"✓ Found: {result['title']} by {result['author_full']}")
            return result
        else:
            print("✗ No books found")
            return None
            
    except Exception as e:
        print(f"✗ Error querying Open Library: {e}")
        return None

def detect_text_vision(image_frame):
    """
    Send image to Google Vision API and get text
    """
    print("\nSending image to Vision API...")
    
    # Encode frame as JPEG
    success, encoded_image = cv2.imencode('.jpg', image_frame)
    if not success:
        print("✗ Failed to encode image")
        return None
    
    # Create Vision API image object
    image = vision.Image(content=encoded_image.tobytes())
    
    # Perform text detection
    try:
        response = vision_client.text_detection(image=image)
        
        if response.error.message:
            print(f"Vision API error: {response.error.message}")
            return None
        
        texts = response.text_annotations
        
        if not texts:
            print("No text detected")
            return None
        
        # First annotation is the full text
        full_text = texts[0].description
        print(f"✓ Detected text:\n{full_text}")
        
        return {
            'full_text': full_text,
            'text_annotations': texts
        }
        
    except Exception as e:
        print(f"✗ Vision API call failed: {e}")
        return None

def parse_book_info(detected_text):
    """
    Parse Vision API response to extract book info
    """
    if not detected_text:
        return None
    
    full_text = detected_text['full_text']
    lines = [line.strip() for line in full_text.split('\n') if line.strip()]
    
    print(f"\nParsing {len(lines)} lines of text...")
    
    # Simple heuristic: 
    # - Look for call number pattern (FIC XXX or numbers)
    # - Assume title is the longest line
    # - Author hint might be in call number
    
    call_number = None
    title_hint = None
    author_hint = None
    
    for line in lines:
        # Check if it's a fiction call number (FIC XXX)
        if line.startswith('FIC') and len(line.split()) >= 2:
            call_number = line
            author_hint = line.split()[1][:3]  # First 3 letters
            print(f"  Found fiction call number: {call_number}")
            print(f"  Author hint: {author_hint}")
        
        # Check if it's a dewey decimal (starts with numbers)
        elif line and line[0].isdigit() and '.' in line[:10]:
            call_number = line
            print(f"  Found dewey decimal: {call_number}")
    
    # Find longest line as potential title
    if lines:
        title_hint = max(lines, key=len)
        print(f"  Title hint: {title_hint}")
    
    return {
        'call_number': call_number,
        'title_hint': title_hint,
        'author_hint': author_hint
    }

# Main Program
print("\n" + "="*50)
print("Single Book Scanner Test")
print("="*50)

print("\nOpening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera!")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera ready!")
print("\n📷 Press 's' to scan a book | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Book Scanner - Press S to scan', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        print("\n" + "="*50)
        print("SCANNING BOOK")
        print("="*50)
        
        # Step 1: Detect text with Vision API
        detected = detect_text_vision(frame)
        
        if detected:
            # Step 2: Parse the detected text
            parsed = parse_book_info(detected)
            
            if parsed:
                # Step 3: Look up full book info
                book_info = lookup_book_openlibrary(
                    title_hint=parsed['title_hint'],
                    author_hint=parsed['author_hint']
                )
                
                if book_info:
                    # Step 4: Display results
                    print("\n" + "="*50)
                    print("BOOK INFORMATION")
                    print("="*50)
                    print(f"Title: {book_info['title']}")
                    print(f"Author: {book_info['author_full']}")
                    print(f"Published: {book_info['first_publish_year']}")
                    print(f"ISBN: {book_info['isbn']}")
                    if parsed['call_number']:
                        print(f"Call Number: {parsed['call_number']}")
                    print("="*50)
                    
                    # Draw on frame
                    result_frame = frame.copy()
                    y_pos = 30
                    cv2.putText(result_frame, f"Title: {book_info['title']}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_pos += 30
                    cv2.putText(result_frame, f"Author: {book_info['author_full']}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow('Result', result_frame)
                    cv2.waitKey(5000)  # Show for 5 seconds
                    cv2.destroyWindow('Result')
        
        print("\n✓ Scan complete\n")
    
    elif key == ord('q'):
        print("\nShutting down...")
        break

cap.release()
cv2.destroyAllWindows()
print("Goodbye!")