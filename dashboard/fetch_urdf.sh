#!/usr/bin/env bash
# Fetch the turtlebot3 URDF + meshes into dashboard/public/urdf/ (idempotent).
# The high-poly STLs (~32 MB) are gitignored, so this needs the network once; skipped if present.
# Source: KweonTJ/Turtlebot3_Platooning (Apache-2.0; ROBOTIS turtlebot3_description /
# turtlebot3_manipulation_description). Leader = waffle_pi + OpenManipulator, follower = waffle.
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
# manipulation.urdf references meshes by an absolute file:/// path; rewrite to package:// (resolved by urdf-loader's packages map).
curl -sSf "$MANI/urdf/turtlebot3_manipulation.urdf" \
 | sed -E 's#file://[^"]*/meshes/#package://turtlebot3_manipulation_description/meshes/#g' \
 > "$SENTINEL"
echo "[urdf] 완료: $OUT"
