# Training Data Distribution

## Overview

| Metric | Value |
|--------|-------|
| **Total Demonstrations** | 1,000 |
| **HDF5 Files** | 20 |
| **Demos per File** | 50 |
| **Training Demos** | 900 (18 files) |
| **Validation Demos** | 100 (2 files) |

## Task Suites

### libero_spatial (10 tasks, 500 demos)
Varying spatial positions of a black bowl to pick and place onto a plate.

| Task | File | Demos |
|------|------|-------|
| 1 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5 | 50 |
| 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 | 50 |
| 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5 | 50 |
| 4 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo.hdf5 | 50 |
| 5 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 | 50 |
| 6 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo.hdf5 | 50 |
| 7 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5 | 50 |
| 8 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5 | 50 |
| 9 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5 | 50 |
| 10 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5 | 50 |

### libero_object (10 tasks, 500 demos)
Varying objects to pick and place into a basket.

| Task | File | Demos |
|------|------|-------|
| 1 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 | 50 |
| 2 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5 | 50 |
| 3 | pick_up_the_butter_and_place_it_in_the_basket_demo.hdf5 | 50 |
| 4 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5 | 50 |
| 5 | pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5 | 50 |
| 6 | pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5 | 50 |
| 7 | pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5 | 50 |
| 8 | pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5 | 50 |
| 9 | pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5 | 50 |
| 10 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5 | 50 |

## Sample Extraction

| Metric | Value |
|--------|-------|
| **Training Samples** | 109,127 |
| **Validation Samples** | 11,630 |
| **Total Samples** | 120,757 |
| **Avg Samples per Demo** | ~109 |

Samples are action chunks extracted with overlapping windows:
- **Chunk size**: 16 timesteps
- **Temporal stride**: Configurable (default extracts at every timestep)

## Data Split

- **Train/Val Split**: 90/10 by file (18 train files, 2 val files)
- Files are shuffled before splitting to ensure task diversity in both sets

## Goal Format

6-dimensional goal vector:
- `pick_pos` (3): Position of object to pick
- `place_pos` (3): Position of place target

Goals are extracted from oracle object positions in demonstrations.
