import numpy as np
import os

ANALYSIS_DATA_DIR = "videos/RoboCasaStackCan-v1"


def analysis():

    episode_names = ["wipe_table_f2b","flip_calendar_page","pull_paper_on_id_card_jsh",
                        "put_object_into_cabinet","flowering_fake_2","put_ice_scoop_in_box_tidy_table",
                        "pour_water_bottle2cup_clean","turn_on_light","place_cube_in_the_center_2",
                        "pick_chips_to_box","wipe_glass_water","pick_pen_on_notebook",
                        "insert_book_shelf","plug_charger","collect_pen","stack_letter_brick",
                        "push_min_chip"]

    data = {x: [] for x in episode_names}
    error = {x: {"instruction":None,"mse":None,"epsiode_num":0,"error_ratio":None} for x in episode_names}

    for root, _, files in os.walk(ANALYSIS_DATA_DIR,followlinks=True):
        for file in files:
            for episode_name in episode_names:
                if file.endswith(".npz") and not file.endswith("dataset_data.npz") and episode_name in file :
                    data[episode_name].append(np.load(os.path.join(root, file)))

    for episode_name in episode_names:
        target_actions=[]
        real_actions=[]
        mse_actions=[]
        error_ratio=[]
        
        for index in range (len(data[episode_name])):
            target_actions.append(data[episode_name][index]["target_actions"])
            real_actions.append(data[episode_name][index]["real_actions"])
            mse_actions.append(np.mean(np.abs(data[episode_name][index]["target_actions"] - data[episode_name][index]["real_actions"]), axis=0))
            error_ratio.append(np.mean(np.abs((data[episode_name][index]["target_actions"]-data[episode_name][index]["real_actions"])
                                              /(np.max(data[episode_name][index]["target_actions"],axis=0)-np.min(data[episode_name][index]["target_actions"],axis=0))),axis=0))
        
        error[episode_name]["instruction"] = data[episode_name][0]["instruction"]
        error[episode_name]["mse"] = np.mean(np.array(mse_actions), axis=0)
        error[episode_name]["mse_mean"] = np.mean(np.abs(error[episode_name]["mse"]))
        error[episode_name]["error_ratio"] = np.mean(np.array(error_ratio), axis=0)
        error[episode_name]["episode_num"] = len(data[episode_name])

    for name in episode_names:
        print()
        print("episode: ", name)
        print("instruction: ",error[name]["instruction"])
        print("episode_num: ", error[name]["episode_num"])
        print("mse_mean: ", error[name]["mse_mean"])
        print()
        print("mse_actions\n", error[name]["mse"])
        print()
        print("error_ratio\n", error[name]["error_ratio"])
        print()
        print("------------------------------------------------------------------------------")


if __name__ == "__main__":
    analysis()