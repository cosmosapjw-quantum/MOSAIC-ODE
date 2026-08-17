import ast,json
from pathlib import Path
def test_mosaic_ode_facade_and_identity():
 import mosaic_ode,weaveode
 assert mosaic_ode.__project_name__=='MOSAIC-ODE';assert mosaic_ode.WeaveBDF1Solver is weaveode.WeaveBDF1Solver;assert mosaic_ode.__version__.startswith('0.2')
def test_bootstrap_manifest_declares_integrated_core():
 data=json.loads(Path('bootstrap/manifest.json').read_text());assert data['project']=='MOSAIC-ODE';assert data['mainline_scope']=='ODE';assert data['dae_scope']=='research';assert {'topology','eess','homotopy','online-learning','gpu-speculation'}<=set(data['core_modules'])
def test_distribution_declares_required_runtime_dependencies():
 module=ast.parse(Path('setup.py').read_text());calls=[n for n in ast.walk(module) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='setup'];keywords={kw.arg:kw.value for kw in calls[0].keywords if kw.arg};deps=set(ast.literal_eval(keywords['install_requires']));assert any(d.startswith('numpy') for d in deps);assert any(d.startswith('scipy') for d in deps);assert any(d.startswith('torch') for d in deps)
