from src.generate_cube import cube_fan, cube_vc
from src.display import run_display_demo

if __name__ == "__main__":
    n = 3
    run_display_demo(cube_fan(n), cube_vc(n))
