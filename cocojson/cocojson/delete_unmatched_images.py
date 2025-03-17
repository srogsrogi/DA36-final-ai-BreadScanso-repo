import os
import json
from pathlib import Path


def remove_images_not_in_json(json_path, image_dir):
    """
    COCO format JSON 파일을 기준으로 불필요한 이미지들을 삭제합니다.

    Args:
        json_path (str): COCO format JSON 파일 경로
        image_dir (str): 이미지 디렉토리 경로
    """
    # JSON 파일 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # JSON의 images 배열에서 파일명 추출
    json_image_files = set()
    for img in coco_data.get('images', []):
        if 'file_name' in img:
            json_image_files.add(img['file_name'])

    if not json_image_files:
        print("경고: JSON 파일에서 이미지 정보를 찾을 수 없습니다!")
        return

    print(f"JSON 파일 내 이미지 수: {len(json_image_files)}")

    # 실제 이미지 디렉토리의 파일 검사
    to_remove = []
    kept = []
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            if filename not in json_image_files:
                to_remove.append(filename)
            else:
                kept.append(filename)

    # 삭제 전 정보 출력
    print(f"\n유지될 이미지 수: {len(kept)}")
    print("유지될 이미지들:")
    for f in kept:
        print(f"- {f}")

    print(f"\n삭제될 이미지 수: {len(to_remove)}")
    if to_remove:
        print("삭제될 이미지들:")
        for f in to_remove:
            print(f"- {f}")

        # 확인 후 삭제 진행
        confirm = input("\n위 파일들을 삭제하시겠습니까? (yes/no): ")
        if confirm.lower() == 'yes':
            for filename in to_remove:
                try:
                    os.remove(os.path.join(image_dir, filename))
                    print(f"삭제됨: {filename}")
                except Exception as e:
                    print(f"삭제 실패 ({filename}): {e}")
            print("\n삭제 완료!")
        else:
            print("삭제가 취소되었습니다.")
    else:
        print("삭제할 파일이 없습니다.")


# 실행
print("처리 시작...")
json_path = "D:\\workspaces\\breadscanso\\cojson\\valid\\_annotations.coco.json"
image_dir = "D:\\workspaces\\breadscanso\\cojson\\valid"



print("데이터 처리:")
remove_images_not_in_json(json_path, image_dir)
