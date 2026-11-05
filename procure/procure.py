import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from procure.cli import main
sys.exit(main())
