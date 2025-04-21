import gc
import spotd.main_code_sec1 as spotae

def train_model(cell_key):
    print("-------------S^2POTAE Model Training-----------------")
    spotae.main(cell_key)

if __name__ == '__main__':
    cell_key = 'cell_type'
    train_model( cell_key)



