import os
import json
import cv2
from pathlib import Path


def yolo_to_coco(yolo_dir, image_dir, output_json):
    if not output_json.endswith('.json'):
        output_json += '.json'

    images = []
    annotations = []
    ann_id = 1  # ann_id를 함수 시작 부분으로 이동

    categories = [  # : 를 = 로 수정하고 categories 리스트 들여쓰기 수정
        {
            "id": 1,
            "name": "bagel",
            "supercategory": "none"
        },
        {
            "id": 2,
            "name": "croissant",
            "supercategory": "none"
        },
        {
            "id": 3,
            "name": "custardcreambread",
            "supercategory": "none"
        },
        {
            "id": 4,
            "name": "pizzabread",
            "supercategory": "none"
        },
        {
            "id": 5,
            "name": "redbeanbread",
            "supercategory": "none"
        },
        {
            "id": 6,
            "name": "saltbread",
            "supercategory": "none"
        },
        {
            "id": 7,
            "name": "soboro",
            "supercategory": "none"
        },
        {
            "id": 8,
            "name": "whitebread",
            "supercategory": "none"
        }
    ]

    # 이미지 확장자 목록
    img_extensions = {'.jpg'}

    # 레이블 파일 목록을 정렬하여 처리
    txt_files = sorted([f for f in os.listdir(yolo_dir) if f.endswith('.txt')])

    for txt_file in txt_files:
        base_name = os.path.splitext(txt_file)[0]

        # 가능한 모든 이미지 확장자에 대해 검사
        img_file = None
        for ext in img_extensions:
            temp_img_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(temp_img_path):
                img_file = base_name + ext
                img_path = temp_img_path
                break

        if img_file is None:
            print(f"Warning: No matching image found for {txt_file}")
            continue

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to read image {img_path}")
                continue

            height, width, _ = img.shape
            image_id = len(images) + 1

            images.append({
                "id": image_id,
                "file_name": img_file,
                "width": width,
                "height": height
            })

            with open(os.path.join(yolo_dir, txt_file), "r") as f:
                lines = f.readlines()

            for line in lines:
                try:
                    data = list(map(float, line.strip().split()))
                    category_id = int(data[0]) + 1

                    if len(data) == 5:  # BBox only
                        x_center, y_center, w, h = data[1:]
                        xmin = max(0, (x_center - w / 2) * width)
                        ymin = max(0, (y_center - h / 2) * height)
                        xmax = min(width, (x_center + w / 2) * width)
                        ymax = min(height, (y_center + h / 2) * height)
                        segmentation = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]

                    elif len(data) > 5:  # Polygon data
                        coords = data[1:]
                        segmentation = []
                        poly = []
                        for i in range(0, len(coords), 2):
                            x = max(0, min(width, coords[i] * width))
                            y = max(0, min(height, coords[i + 1] * height))
                            poly.extend([x, y])
                        segmentation.append(poly)

                        xmin = min(poly[0::2])
                        ymin = min(poly[1::2])
                        xmax = max(poly[0::2])
                        ymax = max(poly[1::2])

                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                        "area": (xmax - xmin) * (ymax - ymin),
                        "segmentation": segmentation,
                        "iscrowd": 0
                    })
                    ann_id += 1

                except ValueError as e:
                    print(f"Warning: Invalid data in {txt_file}: {line.strip()}")
                    continue

        except Exception as e:
            print(f"Error processing {txt_file}: {str(e)}")
            continue

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)


    print(f"✅ 변환 완료: {output_json}")
    print(f"총 이미지 수: {len(images)}")
    print(f"총 어노테이션 수: {len(annotations)}")


# 실행
label_path = "D:\\workspaces\\breadscanso\\cojson\\valid\\labels"
img_path = "D:\\workspaces\\breadscanso\\cojson\\valid\\images"
output_path = "D:\\workspaces\\breadscanso\\cojson\\output\\valid_.coco.json"



yolo_to_coco(label_path, img_path, output_path)


