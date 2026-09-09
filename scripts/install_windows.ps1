# pip install -r requirements.txt fails on Windows: hopsworks -> pyjks -> twofish,
# and twofish has no Windows wheel. Installs everything except twofish (unused
# on the plain JKS auth path this project needs).
# Run from the project root: powershell -File scripts\install_windows.ps1

$ErrorActionPreference = "Stop"

pip install "hopsworks[python]==5.0.6" --no-deps
pip install pyhumps==1.6.1 requests furl boto3 "pandas[mysql]<2.4.0" `
    "numpy<2.5.0,>=1.26.3" mock "avro==1.12.0" "PyMySQL[rsa]" tzlocal `
    "fsspec<2025.12.0" retrying "hopsworks_aiomysql[sa]==0.2.2" `
    "opensearch-py<=2.4.2,>=1.1.0" tqdm "grpcio<2.0.0,>=1.49.1" `
    "protobuf<5.0.0,>=4.25.4" packaging "hopsworks-apigen<2.0.0,>=1.0.4" `
    "click>=8.1" "tomli-w>=1.0" build "pyarrow>=17.0" `
    "confluent-kafka<=2.11.1" "fastavro<=1.12.0,>=1.4.11" "httpx<=0.28.1"

pip install pyjks --no-deps
pip install javaobj-py3 pyasn1 pyasn1-modules pycryptodomex

pip install xgboost scikit-learn python-dotenv holidays entsoe-py

Write-Host "Done. 'pip check' will still complain that pyjks wants twofish - that's expected and harmless."
