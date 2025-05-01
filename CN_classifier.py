import numpy as np
import joblib
import argparse

def main(args):
    metal = args.metal
    xanes = np.load(f'outputs/{metal.lower()}_xanes_uncond/analysis/xanes_conditioning.npy')
    xanes = xanes[:,::2]
    print(xanes.shape)
    loaded_model = joblib.load(f'outputs/CN_classifier/{metal}_CN_rf_model.joblib')
    y_pred = loaded_model.predict(xanes)
    np.save(f'outputs/CN_classifier/{metal}_CN_rf_model_pred.npy', y_pred)
    print(y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metal', type=str, default="Fe",
                        help='Specify central metal type')
    args = parser.parse_args()
    main(args)