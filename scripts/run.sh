#conda activate facesys
uvicorn app.vms:app --host 0.0.0.0 --port 18555 --reload
