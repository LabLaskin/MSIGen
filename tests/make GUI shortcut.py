import sys
from pathlib import Path
from win32com.client import Dispatch

python_path = Path(sys.executable)

env_idxs = []
for i, part in enumerate(python_path.parts):
    if part == 'envs':
        env_idxs.append(i)
if env_idxs:
    env_idxs = env_idxs[-1]
    conda_path = Path(*python_path.parts[:env_idxs])
    env_path = Path(*python_path.parts[:env_idxs+2])
    env_name = env_path.parts[-1]
    MSIGen_path = Path(env_path, r"Lib\site-packages\MSIGen")


else:
    conda_path = Path(*python_path.parts[:-1])
    env_name = ''
    MSIGen_path = Path(conda_path, r"Lib\site-packages\MSIGen")

# paths that will get used in the batch files and shortcut
python_path = str(python_path)
base_activate_path = str(Path(conda_path, r"Scripts\activate.bat"))
MSIGen_GUI_path = str(Path(MSIGen_path, "GUI.py"))
MSIGen_internal_bat_path = str(Path(MSIGen_path,"msigen_gui_internal.bat"))
MSIgen_bat_path = str(Path(MSIGen_path,"msigen_gui.bat"))
MSIGen_shortcut_path = str(Path.home() / r"Desktop\MSIGen GUI.lnk")

# write bat that opens internal bat hidden
with open(MSIgen_bat_path, 'w') as bat_file:
        bat_file.writelines('\n'.join(['@echo off',
            '''powershell "start .\msigen_gui_internal.bat -WindowStyle Hidden"'''
            ]))


# write internal bat that runs GUI    
with open(MSIGen_internal_bat_path, 'w') as bat_file:
        bat_file.writelines('\n'.join(["@echo off",
            "HideCommandWindow",
            f"""call "{base_activate_path}" {env_name}""",
            f""""{python_path}" "{MSIGen_GUI_path}" """,]))

shell = Dispatch('WScript.Shell')
shortcut = shell.CreateShortCut(MSIGen_shortcut_path)
shortcut.Targetpath = MSIgen_bat_path
shortcut.WorkingDirectory = str(Path(MSIgen_bat_path).parent)
shortcut.save()






