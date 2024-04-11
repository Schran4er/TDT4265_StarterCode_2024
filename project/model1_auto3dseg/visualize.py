import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import nrrd


def plot_by_z_slice_idx(z, lbl, pred, name):

    img_slice = lbl[:, :, z] == 0
    label_slice = lbl[:, :, z] == 1
    background_slice = pred[:, :, z, 0] if pred.ndim == 4 else pred[:, :, z] == 0
    foreground_slice = pred[:, :, z, 1] if pred.ndim == 4 else pred[:, :, z] == 1

    plt.subplot(2, 2, 1)
    plt.imshow(img_slice)
    plt.title("background groundtruth")
    cbar = plt.colorbar(shrink=0.8)
    plt.subplot(2, 2, 2)
    plt.imshow(label_slice)
    plt.title("foreground_groundtruth")
    cbar = plt.colorbar(shrink=0.8)
    plt.subplot(2, 2, 3)
    plt.imshow(background_slice)
    plt.title("background prediction")
    cbar = plt.colorbar(shrink=0.8)
    plt.subplot(2, 2, 4)
    plt.imshow(foreground_slice)
    plt.title("foreground prediction")
    cbar = plt.colorbar(shrink=0.8)
    # set the spacing between subplots
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    path = f"/cluster/work/felixzr/TDT4265_StarterCode_2024/project/model1_auto3dseg/output_dir/plot_{name}.png"
    plt.savefig(path)


def main():
    # img_nib = nib.load(os.path.join(dataroot_dir, sim_datalist["testing"][0]["image"]))
    # lbl_nib = nib.load(os.path.join(dataroot_dir, sim_datalist["testing"][0]["label"]))
    # img = np.array(img_nib.dataobj)
    # lbl = np.array(lbl_nib.dataobj)


    # nrrd_data, nrrd_header = nrrd.read(os.path.join(dataroot_dir, sim_datalist["testing"][0]["image"]))
    # img = np.array(nrrd_data)
    # nrrd_data, nrrd_header = nrrd.read(os.path.join(dataroot_dir, sim_datalist["testing"][0]["label"]))

    # TODO: adjust color map
    truth_path = "/cluster/work/felixzr/TDT4265_StarterCode_2024/data_flat/Annotation_Normal_17.nrrd"
    nrrd_data, nrrd_header = nrrd.read(truth_path)
    lbl = np.array(nrrd_data)

    
    prediction_path = "/cluster/work/felixzr/TDT4265_StarterCode_2024/project/model1_auto3dseg/segresnet_0/prediction_testing/CTCA_Normal_17.nii.gz"
    # image_name = sim_datalist["testing"][0]["image"].split(".")[0]
    # prediction_nib = nib.load(os.path.join(work_dir, "ensemble_output", image_name + "_ensemble" + ".nii.gz"))
    prediction_nib = nib.load(prediction_path)
    pred = np.array(prediction_nib.dataobj)

    for i in range(100, 200, 10):
        plot_by_z_slice_idx(i, lbl, pred, i)


if __name__ == '__main__':
    main()