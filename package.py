"""Build the final submission zip from a chosen version and run the organizers' zip check.

    python package.py sub8

Writes final/main.py and submission.zip (main.py at the archive root, nothing else),
then runs public/tools/check_submission_zip.py on the zip exactly as ingestion would.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(HERE, sys.argv[1], "main.py")
final = os.path.join(HERE, "final")
os.makedirs(final, exist_ok=True)
for name in os.listdir(final):
    os.remove(os.path.join(final, name))
shutil.copyfile(src, os.path.join(final, "main.py"))

zpath = os.path.join(HERE, "submission.zip")
if os.path.exists(zpath):
    os.remove(zpath)
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(final, "main.py"), arcname="main.py")
with zipfile.ZipFile(zpath) as z:
    print("zip contents:", z.namelist())
print("sha256 main.py:", hashlib.sha256(open(os.path.join(final, "main.py"), "rb").read()).hexdigest()[:16])
print("zip size: %d bytes" % os.path.getsize(zpath))

env = dict(os.environ, SB_TOTAL_BUDGET="120", SB_VERBOSE="0")
done = subprocess.run([sys.executable, os.path.join(HERE, "public", "tools", "check_submission_zip.py"), zpath],
                      cwd=os.path.join(HERE, "public"), capture_output=True, text=True, env=env)
print("check_submission_zip:", (done.stdout + done.stderr).strip()[-600:], "| exit", done.returncode)
