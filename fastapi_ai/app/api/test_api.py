import requests
import json
import time

API_URL = "http://localhost:8000"

def test_health():
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_ocr():
    print("\nTesting OCR endpoint...")
    
    # Sample request
    payload = {
        "image_id": "test-001",
        "user_id": "test-user",
       "image_url": "http://localhost:9000/image1.png"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    start = time.time()
    response = requests.post(
        f"{API_URL}/api/ocr/extract-metrics",
        json=payload,
        timeout=120
    )
    elapsed = time.time() - start
    
    print(f"\nStatus: {response.status_code}")
    print(f"Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nResult:")
        print(f"  Status: {result['status']}")
        print(f"  Metrics: {len(result['extracted_metrics'])}")
        print(f"  Processing Time: {result['processing_time_seconds']:.2f}s")
        print(f"  Engine: {result['ocr_engine']}")
        print(f"\nMetadata:")
        print(json.dumps(result['metadata'], indent=2))
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

if __name__ == "__main__":
    print("=" * 50)
    print("API Test Suite")
    print("=" * 50)
    
    # Test health
    health_ok = test_health()
    
    if health_ok:
        # Test OCR
        ocr_ok = test_ocr()
        
        print("\n" + "=" * 50)
        print("Test Results:")
        print(f"  Health: {'PASS' if health_ok else '✗ FAIL'}")
        print(f"  OCR:    {' PASS' if ocr_ok else '✗ FAIL'}")
        print("=" * 50)
    else:
        print("\n✗ Health check failed - skipping OCR test") 