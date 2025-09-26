# TODO: Fix OCR Issues in GeoAdhikar App

## Steps to Complete

- [x] Download eng.traineddata to ./tessdata
- [x] Download hin.traineddata to ./tessdata
- [x] Update app.py to set TESSDATA_PREFIX to ./tessdata if it exists
- [x] Update requirements.txt to fix google-generativeai version
- [ ] Test the OCR functionality by running the app and uploading a PDF
- [ ] Check logs for any remaining errors

## Progress

- Created tessdata directory
- Downloaded hin.traineddata
- Downloaded eng.traineddata
- Updated requirements.txt for google-generativeai
- Updated app.py to use local tessdata
