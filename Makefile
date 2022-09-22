train:
	python src/train.py

convert:
	python src/convert_onnx.py

test-onnx:
	python src/test_onnx.py

build-bin:
	pyinstaller --onefile --distpath build --workpath tmp src/build.py