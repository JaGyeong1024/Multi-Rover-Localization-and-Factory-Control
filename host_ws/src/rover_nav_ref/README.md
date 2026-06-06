# rover_nav_ref — 로버측 **인터페이스 레퍼런스** (팀원 B용), host 런타임/mock 아님

호스트(`host_mission`)가 내리는 명령을 로버가 **어떤 인터페이스로 받고 되돌려주는지**만 보여주는
얇은 레퍼런스다. 실제 **경로계획(lane 그래프 라우팅) + cmd_vel 제어는 팀원 B(로버 플래닝·제어) 몫**이라
`rover_nav_node.py`에 **`TODO(teammate B)`** 로 비워뒀다 — 채워 넣을 자리.

> 적용 가이드(받은 mode별 행동·feedback state·deadman) = **`docs/rover_interface_contract.md`** (SSOT).
> 데모 검증용 동작 mock(가짜 pose 발행·이동 시뮬)은 **로버 코드가 아니라 host 테스트 도구** →
> `host_mission/mock_rover_node` (이 패키지엔 두지 않는다).

## 계약 (host ↔ rover)
| 토픽 | 방향 | 타입 |
|---|---|---|
| `/{robot}/target_cmd` | host→로버 구독 | `host_mission_interfaces/TargetCommand` (target_node + mode) |
| `/{robot}/global/position` | host localization→로버 구독 | `geometry_msgs/PoseStamped` (= 로버 pose 입력) |
| `/{robot}/nav_feedback` | 로버 발행→host | `host_mission_interfaces/NavFeedback` (state, cmd_id echo) |
| `/{robot}/cmd_vel` | 로버 내부 | `geometry_msgs/Twist` (팀원 B 컨트롤러 출력) |

deadman: `target_cmd` `header.stamp`가 0.5s stale → STOP.

## 실행 (인터페이스 확인용 — 이동은 안 함)
```
ros2 run rover_nav_ref rover_nav_node --ros-args -p robot:=leader
```
`target_cmd`를 받아 mode별로 feedback state를 보고하는 골격만 동작. 실제 주행은 TODO를 채운 뒤.
