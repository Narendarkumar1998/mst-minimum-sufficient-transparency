# MST SARL multi-agent execution (MST_SARL_Stable)

Event-driven SARL implementation of Minimum Sufficient Transparency used for
the 36-condition execution reported in the paper.

## Reconstruction notice

The original source folder of this project was lost after the experiment was
run. This project is a **reconstruction** made from the documented design
(the research record, the candidate pools listed in Online Resource 1, and
source lines preserved in the source-to-result audit files). The Java core
(`MSTService`, `MSTResult`, `MSTSweepRecorder`) has been verified to reproduce
the original stored output `reference-output/sarl_mst_multiscenario_sweep.csv`
**exactly**: all 36 conditions and all 576 values are identical, including
every floating-point digit. The SARL agents follow the documented event flow
and reuse the identifiers preserved from the original source.

## Architecture

| Layer | File | Responsibility |
|---|---|---|
| Java core | `MSTService.java` | Candidate pools; enumerates the 7 non-empty subsets, rejects inadmissible ones, returns the minimum-objective sufficient subset or INFEASIBLE |
| Java core | `MSTResult.java` | Immutable result of one evaluation |
| Java core | `MSTSweepRecorder.java` | CSV output, verification contract, nestedness audit |
| Java core | `MSTSweepMain.java` | Plain-Java run of the same sweep without agents (for checking) |
| SARL events | `MSTEvents.sarl` | `RoleReady`, `MSTRequest`, `MSTResponse`, `StopRoles` |
| SARL agent | `RoleAgent.sarl` | Acts for one role; calls `MSTService.evaluate(role, scenario, tau)` and emits `MSTResponse` |
| SARL agent | `MSTBootAgent.sarl` | Spawns the three role agents, runs the scenario and threshold loops, records results, stops the system |

Event flow: `MSTBootAgent -> MSTRequest -> RoleAgent -> MSTService -> MSTResponse -> MSTSweepRecorder`.

Selection rules: balanced objective J = (P + C_norm + K) / 3, with interpretation
cost normalized by the complete three-item pool and K = |S| / 3; objectives
within 1e-10 are broken in favour of fewer items. A feasible row is verified
when it is sufficient, admissible and deletion-minimal; an infeasible row when
it is insufficient and selects nothing.

## Requirements

- SARL IDE or SARL SDK 0.15.1
- JDK 21
- Maven (dependencies are downloaded on first build)

## Run

In the SARL IDE: import as an existing Maven project, build it, then run
`src/main/sarl/org/mst/sarl/agents/MSTBootAgent.sarl` as a SARL Application.
The expected final line is:

```
MULTI-SCENARIO SWEEP COMPLETE | conditions=36 | verified=36/36 | nested=17/17
```

The output is written to `results/sarl_mst_multiscenario_sweep.csv`. Rows within
one threshold may appear in a different order between runs, because the three
role agents answer concurrently; the content per condition is identical.

To check the Java core without SARL:

```bash
javac -d out src/main/java/org/mst/sarl/core/*.java
java -cp out org.mst.sarl.core.MSTSweepMain results/sarl_mst_multiscenario_sweep.csv
```
