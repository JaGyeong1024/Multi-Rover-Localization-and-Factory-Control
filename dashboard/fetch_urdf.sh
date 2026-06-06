#!/usr/bin/env bash
# turtlebot3 URDF + 메시를 rover 레포에서 dashboard/public/urdf/ 로 가져온다 (idempotent).
# 고폴리 STL(~32MB)이라 git엔 안 올림(.gitignore) → 최초 1회만 네트워크. 이미 있으면 스킵.
# 출처: KweonTJ/Turtlebot3_Platooning (Apache-2.0; ROBOTIS turtlebot3_description /
#       turtlebot3_manipulation_description). 리더=waffle_pi+OpenManipulator, 팔로워=waffle.
set -e
cd "$(dirname "$0")"
OUT=public/urdf
SENTINEL="$OUT/turtlebot3_manipulation_description/urdf/turtlebot3_manipulation.urdf"
[ -f "$SENTINEL" ] && { echo "[urdf] 이미 있음 — 스킵 ($OUT)"; exit 0; }

RAW=https://raw.githubusercontent.com/KweonTJ/Turtlebot3_Platooning/main
DESC=$RAW/turtlebot3/turtlebot3_description
MANI=$RAW/turtlebot3_manipulation/turtlebot3_manipulation_description
DOUT=$OUT/turtlebot3_description
MOUT=$OUT/turtlebot3_manipulation_description
mkdir -p "$DOUT/urdf" "$DOUT/meshes/bases" "$DOUT/meshes/wheels" "$DOUT/meshes/sensors" \
         "$MOUT/urdf" "$MOUT/meshes/turtlebot3_waffle_pi" "$MOUT/meshes/open_manipulator_x"

echo "[urdf] 메시 다운로드 (~32MB, 최초 1회)..."
curl -sSf "$DESC/meshes/bases/waffle_base.stl" -o "$DOUT/meshes/bases/waffle_base.stl"
curl -sSf "$DESC/meshes/wheels/left_tire.stl"  -o "$DOUT/meshes/wheels/left_tire.stl"
curl -sSf "$DESC/meshes/wheels/right_tire.stl" -o "$DOUT/meshes/wheels/right_tire.stl"
curl -sSf "$DESC/meshes/sensors/lds.stl"       -o "$DOUT/meshes/sensors/lds.stl"
for m in base left_tire right_tire; do
  curl -sSf "$MANI/meshes/turtlebot3_waffle_pi/$m.stl" -o "$MOUT/meshes/turtlebot3_waffle_pi/$m.stl"
done
for m in link1 link2 link3 link4 link5 gripper_left_palm gripper_right_palm; do
  curl -sSf "$MANI/meshes/open_manipulator_x/$m.stl" -o "$MOUT/meshes/open_manipulator_x/$m.stl"
done

echo "[urdf] URDF 다운로드 + 경로 재작성..."
curl -sSf "$DESC/urdf/turtlebot3_waffle.urdf" -o "$DOUT/urdf/turtlebot3_waffle.urdf"
# manipulation.urdf는 file:///<ktj 절대경로> 메시참조 → package:// 로 재작성(urdf-loader packages 맵으로 해소).
curl -sSf "$MANI/urdf/turtlebot3_manipulation.urdf" \
 | sed 's#file:///home/ktj/turtlebot3_ws/install/turtlebot3_manipulation_description/share/turtlebot3_manipulation_description/meshes/#package://turtlebot3_manipulation_description/meshes/#g' \
 > "$SENTINEL"
echo "[urdf] 완료: $OUT"
