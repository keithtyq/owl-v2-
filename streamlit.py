import os
import glob
import random
import math
import json
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

@st.cache_data
def load_predictions(results_file="results_cached.json"):
    with open(results_file, "r") as f:
        results = json.load(f)
    pred_map = {}
    for r in results:
        fname = r["image"]
        if fname not in pred_map:
            pred_map[fname] = []
        pred_map[fname].append({
            "bbox": r["box"],  # [x1, y1, x2, y2]
            "label": r["label"],
            "score": r["score"]
        })
    return pred_map

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def get_label_color(label, color_map):
    """Get a consistent color for a label, generating a new one if not in the map."""
    if label not in color_map:
        color_map[label] = tuple(random.randint(0, 255) for _ in range(3))
    return color_map[label]

def draw_boxes_matplotlib(image_path, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores=None):
    """
    Draw GT and predicted boxes using matplotlib, as in your notebook.
    All boxes must be in [x1, y1, x2, y2] format.
    """
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    # Plot GT boxes (green)
    for bbox, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, label, color='g', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Plot predicted boxes (red)
    for i, (bbox, label) in enumerate(zip(pred_boxes, pred_labels)):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        label_str = f"{label}"
        if pred_scores is not None:
            label_str += f" ({pred_scores[i]:.2f})"
        ax.text(x1, y1, label_str, color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def main():
    st.set_page_config(layout="wide")
    nav = st.sidebar.selectbox("SELECT MODULE", 
        ["001-Single Image Viz", 
         "002-Multiple Image Viz",
         "003-View all images",
         "004-Filter images by label"])

    if nav == "001-Single Image Viz":
        df_imported = pd.read_pickle("dataset/20250327_091546_df_all_2.pkl")  
        df_imported = df_imported.reset_index()
        list_filename = df_imported["filename"].unique().tolist()
        img_max_hh = 512

        st.title("001-Single Image Viz")
        val_1 = st.slider("Browse Images", 0, len(list_filename)-1)
        val_2 = st.checkbox("Show bounding boxes", value=True)
        df_subset = df_imported[df_imported["filename"]==list_filename[val_1]]
        filename = df_subset["filename"].iloc[0]
        path = f"dataset/images/{filename}"
        st.text(path)
        img = Image.open(path).convert("RGB")
        img_ww, img_hh = img.size
        col1, col2 = st.columns([0.5, 1])

        with col2:
            st.dataframe(df_subset[["category_id", "label", "bbox", "set", "filename", "id", "image_id"]])

        with col1:
            overlay_option = st.radio(
                "Show boxes:",
                ["Ground Truth only", "Predictions only", "Both (GT + Pred)"],
                horizontal=True
            )
            bboxes = df_subset['bbox'].tolist()
            labels = df_subset['label'].tolist()
            gt_label_set = set(labels)
            pred_map = load_predictions("results_cached.json")
            preds = pred_map.get(filename, [])
            filtered_preds = [p for p in preds if p["label"] in gt_label_set]
            pred_bboxes = [p["bbox"] for p in filtered_preds]
            pred_labels = [p["label"] for p in filtered_preds]
            pred_scores = [p["score"] for p in filtered_preds]
            # Convert GT to [x1, y1, x2, y2]
            gt_boxes_xyxy = [xywh_to_xyxy(b) for b in bboxes]
            # Preds are already [x1, y1, x2, y2]
            pred_boxes_xyxy = pred_bboxes

            if overlay_option == "Ground Truth only":
                image_with_boxes = draw_boxes_matplotlib(
                    path, gt_boxes_xyxy, labels, [], [], None)
            elif overlay_option == "Predictions only":
                image_with_boxes = draw_boxes_matplotlib(
                    path, [], [], pred_boxes_xyxy, pred_labels, pred_scores)
            else:  # Both
                image_with_boxes = draw_boxes_matplotlib(
                    path, gt_boxes_xyxy, labels, pred_boxes_xyxy, pred_labels, pred_scores)

            st.write("(ww,hh):", img.size)
            st.image(image_with_boxes, use_container_width=True)

    #########################################################################
    elif nav == "002-Multiple Image Viz":
        st.title("002-Multiple Image Viz")
        df_imported = pd.read_pickle("dataset/20250327_091546_df_all_2.pkl")    
        df_imported = df_imported.reset_index()
        df_imported["path"] = "dataset/images/" + df_imported["filename"]
        list_path = df_imported["path"].unique().tolist()
        num_images = 10
        num_batches = math.ceil(len(list_path) / num_images)
        batch_index = st.slider("Select batch", 0, max(0, num_batches-1), 0)
        start_idx = batch_index * num_images
        end_idx = min(start_idx + num_images, len(list_path))
        selected_images = list_path[start_idx:end_idx]
        st.write(f"Displaying images {start_idx + 1} to {end_idx} out of {len(list_path)}")
        cols_per_row = 5
        rows = [st.columns(cols_per_row) for _ in range(2)]
        pred_map = load_predictions("results_cached.json")

        for i, img_path in enumerate(selected_images):
            col = rows[i // cols_per_row][i % cols_per_row]
            with col:
                img = Image.open(img_path)
                df_subset = df_imported[df_imported["path"]==img_path]
                bboxes = df_subset["bbox"].tolist()
                labels = df_subset["label"].tolist()
                gt_label_set = set(labels)
                filename = os.path.basename(img_path)
                preds = pred_map.get(filename, [])
                filtered_preds = [p for p in preds if p["label"] in gt_label_set]
                pred_bboxes = [p["bbox"] for p in filtered_preds]
                pred_labels = [p["label"] for p in filtered_preds]
                pred_scores = [p["score"] for p in filtered_preds]
                gt_boxes_xyxy = [xywh_to_xyxy(b) for b in bboxes]
                pred_boxes_xyxy = pred_bboxes
                image_with_boxes = draw_boxes_matplotlib(
                    img_path, gt_boxes_xyxy, labels, pred_boxes_xyxy, pred_labels, pred_scores)
                st.write(img_path)
                st.image(image_with_boxes, use_container_width=True)

    #########################################################################
    elif nav == "003-View all images":
        st.title("003-View all images")
        df_imported = pd.read_pickle("dataset/20250327_091546_df_all_2.pkl")
        df_imported = df_imported.reset_index()
        df_imported["path"] = "dataset/images/" + df_imported["filename"]
        list_path = df_imported["path"].unique().tolist()
        num_images = 300
        num_batches = math.ceil(len(list_path) / num_images)
        batch_index = st.slider("Select batch", 0, max(0, num_batches-1), 0)
        start_idx = batch_index * num_images
        end_idx = min(start_idx + num_images, len(list_path))
        selected_images = list_path[start_idx:end_idx]
        st.write(f"Displaying images {start_idx + 1} to {end_idx} out of {len(list_path)}")
        cols_per_row = 5
        rows = [st.columns(cols_per_row) for _ in range(60)]
        pred_map = load_predictions("results_cached.json")

        for i, img_path in enumerate(selected_images):
            col = rows[i // cols_per_row][i % cols_per_row]
            with col:
                img = Image.open(img_path)
                df_subset = df_imported[df_imported["path"]==img_path]
                bboxes = df_subset["bbox"].tolist()
                labels = df_subset["label"].tolist()
                gt_label_set = set(labels)
                filename = os.path.basename(img_path)
                preds = pred_map.get(filename, [])
                filtered_preds = [p for p in preds if p["label"] in gt_label_set]
                pred_bboxes = [p["bbox"] for p in filtered_preds]
                pred_labels = [p["label"] for p in filtered_preds]
                pred_scores = [p["score"] for p in filtered_preds]
                gt_boxes_xyxy = [xywh_to_xyxy(b) for b in bboxes]
                pred_boxes_xyxy = pred_bboxes
                image_with_boxes = draw_boxes_matplotlib(
                    img_path, gt_boxes_xyxy, labels, pred_boxes_xyxy, pred_labels, pred_scores)
                st.write(img_path)
                st.image(image_with_boxes, use_container_width=True)

    #########################################################################
    elif nav == "004-Filter images by label":
        st.title("004-Filter images by label")
        df_imported = pd.read_pickle("dataset/20250327_091546_df_all_2.pkl")
        list_label = sorted(df_imported["label"].unique().tolist())
        label_chosen = st.sidebar.selectbox("Select Label:", list_label)
        st.write("Label Chosen :", label_chosen)
        df_imported = df_imported[df_imported["label"]==label_chosen]
        df_imported["path"] = "dataset/images/" + df_imported["filename"]
        list_path = df_imported["path"].unique().tolist()
        num_images_label = len(list_path)
        st.write(f"Number of images with label, {label_chosen} : {num_images_label}")
        cols_per_row = 5
        rows = [st.columns(cols_per_row) for _ in range(60)]
        pred_map = load_predictions("results_cached.json")

        for i, img_path in enumerate(list_path):
            col = rows[i // cols_per_row][i % cols_per_row]
            with col:
                img = Image.open(img_path)
                df_subset = df_imported[df_imported["path"]==img_path]
                bboxes = df_subset["bbox"].tolist()
                labels = df_subset["label"].tolist()
                gt_label_set = set(labels)
                filename = os.path.basename(img_path)
                preds = pred_map.get(filename, [])
                filtered_preds = [p for p in preds if p["label"] in gt_label_set]
                pred_bboxes = [p["bbox"] for p in filtered_preds]
                pred_labels = [p["label"] for p in filtered_preds]
                pred_scores = [p["score"] for p in filtered_preds]
                gt_boxes_xyxy = [xywh_to_xyxy(b) for b in bboxes]
                pred_boxes_xyxy = pred_bboxes
                image_with_boxes = draw_boxes_matplotlib(
                    img_path, gt_boxes_xyxy, labels, pred_boxes_xyxy, pred_labels, pred_scores)
                st.write(img_path)
                st.image(image_with_boxes, use_container_width=True)

if __name__ == '__main__':
    main()