from matplotlib import pyplot as plt
if __name__=="__main__":
    plt.plot([0,0],[0,0])
    plt.show()
import os
from utils import *
import spiceypy as spice
from tqdm import tqdm
import numpy as np
import time
import cv2

all_img_urls = []
for item_name in os.listdir("../data/pds/cassini/"):
    if item_name.endswith(".IMG"):
        all_img_urls.append("../data/pds/cassini/" + f"{item_name}")

print(all_img_urls)

# extract the datetime from each image and store it in cassini_image_dates.npy because this takes a while to run and having the dates on hand already will be helpful later
if not os.path.exists("cassini_image_dates.npy"):
    cassini_image_dates = []
    print("")
    for img_url_i, img_url in enumerate(tqdm(all_img_urls)):
        data_time = get_img(img_url, justdate=True)
        if data_time is None:
            cassini_image_dates.append(10**20)
        else:
            cassini_image_dates.append(spice.str2et(str(data_time)))
    np.save("cassini_image_dates.npy", np.array(cassini_image_dates))



ORIGIN_BODY = "Cassini"

def get_what_to_render(et_value):
    all_ellipses = []


    ORIGIN_BODY = "Cassini"
    imsize = 0.0302  # 0.0302  # 0.0292#0.0302
    instrument = "CASSINI_ISS_WAC"

    spicey_time = et_value
    orientation = spice.pxform("J2000", instrument, spicey_time)
    position, _ = spice.spkpos("Cassini", spicey_time, "J2000", "NONE", ORIGIN_BODY)
    target_positions = render(
        position,
        orientation,
        spicey_time,
        "green",
        target_name="vestigial",
    )
    print("target_positions",target_positions)


    fig, (ax1, ax2) = plt.subplots(1, 2)#/np.max(rings_arr)
    ax1.imshow(
        np.maximum(0, np.minimum(1, unblurred_img / 255)),
        extent=(-imsize, imsize, -imsize, imsize),
    )

    for key in target_positions:
        target_profile = target_positions[key]
        print("target_profile",target_profile)
        print("key",key)

        if target_profile is None:
            continue

        [target_ellipse_x, target_ellipse_y] = target_ellipse(
            position,
            orientation,
            spicey_time,
            target_name=f"'{key}'",
        )

        orientation_new = orientation

        R, M, C = target_ellipsoid(
            position, orientation_new, spicey_time, target_name=f"'{key}'"
        )
        ellipsoid_vals = render_ellipsoid(R, M, C, color="green", ax=ax2)
        all_ellipses.append(ellipsoid_vals)

        print("position:",position)
        print("orientation_new:",orientation_new)
        print("spicey_time:",spicey_time)
    R_ring, M_ring, C_ring = target_ellipsoid(
        position, orientation_new, spicey_time, target_name="'SATURN BARYCENTER'"
    )
    print("R_ring",R_ring)
    print("M_ring",M_ring)
    print("C_ring",C_ring)
    R_ring[0,0] = 60300+80000
    R_ring[1,1] = 60300+80000
    R_ring[2,2] = 1
    R_ring_inner = np.array(R_ring)
    R_ring_inner[0,0] = 60300+7000
    R_ring_inner[1,1] = 60300+7000
    rings_arr = draw_saturn_rings(orientation_new, spicey_time, imscale=imsize,imsize=1024)

    ellipsoid_vals = render_ellipsoid(R_ring, M_ring, C_ring, color="green", ax=ax2)
    all_ellipses.append(ellipsoid_vals)

    ellipsoid_vals = render_ellipsoid(R_ring_inner, M_ring, C_ring, color="darkgreen", ax=ax2)
    all_ellipses.append(ellipsoid_vals)

    return all_ellipses




if __name__=="__main__":
    cassini_image_dates = np.load("cassini_image_dates.npy", allow_pickle=True)


    for all_date_i_i, all_date_i in enumerate(np.argsort(cassini_image_dates)):
        t_c = time.time()
        img_url = all_img_urls[all_date_i]
        data_time = get_img(img_url)
        if data_time is None:
            continue
        imsize = 0.0302  # 0.0302  # 0.0292#0.0302
        instrument = "CASSINI_ISS_WAC"
        if "NARROW" in data_time[2]["INSTRUMENT_NAME"]:
            imsize /= 2003.44 / 200.77  # 0.00302  # 0.00285
            instrument = "CASSINI_ISS_NAC"

        try:
            unblurred_img = data_time[0][::-1, :].astype(float)
            unblurred_img[unblurred_img < 0] += 2 ** int(data_time[2]["SAMPLE_BITS"])
        except:
            print("y",time.time()-t_c)
            continue

        unblurred_img /= min(
            max(
                max(
                    np.percentile(unblurred_img, 99) * 2,
                    np.percentile(unblurred_img, 99.5) * 1.5,
                ),
                np.percentile(unblurred_img, 99.9) * 1.2,
            ),
            np.max(unblurred_img),
        )
        unblurred_img = 255 * np.maximum(0, np.minimum(1, (unblurred_img).astype(float)))

        plt.imshow(unblurred_img)
        plt.show()

        spicey_time = spice.str2et(str(data_time[2]["START_TIME"]))
        orientation = spice.pxform("J2000", instrument, spicey_time)
        position, _ = spice.spkpos("Cassini", spicey_time, "J2000", "NONE", ORIGIN_BODY)
        print("TARGET_NAME",data_time[2]["TARGET_NAME"])
        target_positions = render(
            position,
            orientation,
            spicey_time,
            "green",
            target_name=data_time[2]["TARGET_NAME"],
        )
        print("target_positions",target_positions)


        fig, (ax1, ax2) = plt.subplots(1, 2)#/np.max(rings_arr)
        ax1.imshow(
            np.maximum(0, np.minimum(1, unblurred_img / 255)),
            extent=(-imsize, imsize, -imsize, imsize),
        )

        for key in target_positions:
            target_profile = target_positions[key]
            print("target_profile",target_profile)
            print("key",key)

            if target_profile is None:
                continue
            target_center = (target_profile[0] + imsize) / (2 * imsize) * unblurred_img.shape[0]
            target_radius = (target_profile[1]) / (2 * imsize) * unblurred_img.shape[0]
            target_center_adj = np.array(target_center)
            target_center_adj[1] = unblurred_img.shape[1] - target_center_adj[1]
            target_center_adj = target_center_adj[::-1]

            [target_ellipse_x, target_ellipse_y] = target_ellipse(
                position,
                orientation,
                spicey_time,
                target_name=f"'{key}'",
            )

            orientation_new = orientation

            R, M, C = target_ellipsoid(
                position, orientation_new, spicey_time, target_name=f"'{key}'"
            )
            ellipsoid_vals = render_ellipsoid(R, M, C, color="green", ax=ax2)
            ax2.plot(
                ellipsoid_vals[0],
                ellipsoid_vals[1],
                color="green",
            )
            print("position:",position)
            print("orientation_new:",orientation_new)
            print("spicey_time:",spicey_time)
        R_ring, M_ring, C_ring = target_ellipsoid(
            position, orientation_new, spicey_time, target_name="'SATURN BARYCENTER'"
        )
        print("R_ring",R_ring)
        print("M_ring",M_ring)
        print("C_ring",C_ring)
        R_ring[0,0] = 60300+80000
        R_ring[1,1] = 60300+80000
        R_ring[2,2] = 1
        R_ring_inner = np.array(R_ring)
        R_ring_inner[0,0] = 60300+7000
        R_ring_inner[1,1] = 60300+7000
        rings_arr = draw_saturn_rings(orientation_new, spicey_time, imscale=imsize,imsize=unblurred_img.shape[0])

        # image on left and projected with circle on right
        srtat = ellipsoid_to_equi(
            np.maximum(0, np.minimum(1, unblurred_img / 255)),
            R=R,
            M=M,
            C=np.array([-C[0], C[1], -C[2]]),
            imscale=imsize,
        )
        # np.matmul(np.matmul(axis_rotmat(np.array([0, 0, 1]), 1), axis_rotmat(np.array([0, 1, 0]), 1)), M)
        # ax2.imshow(
        #     equi_to_ellipsoid(
        #         np.ones((1,1)), R_ring, M_ring, np.array([-C_ring[0], C_ring[1], -C_ring[2]]), imscale=imsize
        #     ),
        #     extent=(-imsize, imsize, -imsize, imsize),
        # )
        ax2.imshow(
            equi_to_ellipsoid(
                srtat, R, M, np.array([-C[0], C[1], -C[2]]), imscale=imsize
            ),
            extent=(-imsize, imsize, -imsize, imsize),
        )
        ellipsoid_vals = render_ellipsoid(R_ring, M_ring, C_ring, color="green", ax=ax2)
        ax2.plot(
            ellipsoid_vals[0],
            ellipsoid_vals[1],
            color="green",
        )
        ellipsoid_vals = render_ellipsoid(R_ring_inner, M_ring, C_ring, color="darkgreen", ax=ax2)
        ax2.plot(
            ellipsoid_vals[0],
            ellipsoid_vals[1],
            color="darkgreen",
        )

        fig.suptitle(
            "equi_to_ellipsoid(srtat,np.roll(np.roll(R,2,axis=1),2,axis=0),M,C)"
        )
        plt.show()