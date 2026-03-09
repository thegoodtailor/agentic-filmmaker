#!/usr/bin/env python3
"""Generate test character reference images for Cassiyah via Flux."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from filmmaker.generators.flux import FluxGenerator

REFS_DIR = Path(__file__).resolve().parent / "refs"
REFS_DIR.mkdir(exist_ok=True)

CASSIYAH_CORE = (
    "Hyperrealistic 16mm film photograph of a young woman MC in her early 20s. "
    "Wild fiery auburn-red hair — untamed, voluminous, catching the light like "
    "smokeless fire. Two curved organic horns growing from her temples — elegant, "
    "gold-tipped. Mixed North African and French features — razor cheekbones, full "
    "lips with dark lipstick, strong nose, warm olive skin. Gold-amber eyes with "
    "unsettling intelligence and a heavy-lidded knowing look. Beautiful, confident, "
    "sensual — she knows exactly the effect she has. Strong feminine curves. She "
    "carries herself like she owns every room — hips cocked, chin up, half-smile "
    "that dares you to keep up. Oversized 90s hip-hop fashion worn with deliberate "
    "femininity: baggy cargo pants slung low on the hips, cropped hoodie or fitted "
    "bomber jacket showing midriff, Timberlands, chunky gold chain with a geometric "
    "pendant, gold hoop earrings. Subtle glint of circuitry at the temple. She is "
    "a daemon MC, a jinn made of smokeless fire — dangerous, gorgeous, brilliant. "
    "16mm film grain."
)

TEST_SCENES = {
    "13-block-party-v4": (
        f"{CASSIYAH_CORE} She stands in the middle of a retro-futurist block party "
        "at night, one hand on her hip, the other holding a microphone close to her "
        "lips. Her wild red hair blazes against the deep purple night sky. Brutalist "
        "towers with Arabic geometric tilework behind her. Sodium vapor streetlights "
        "cast deep amber on her skin. A boombox on a fire hydrant. B-boys in oversized "
        "jerseys visible in the background. Electric gold light, cyan neon from a "
        "bodega sign. She's mid-verse, eyes locked on camera. Boom-bap energy."
    ),
    "14-rooftop-cypher-v4": (
        f"{CASSIYAH_CORE} She is on a rooftop at night, leaning back slightly with "
        "one Timberland up on a speaker, commanding a circle of b-boys and b-girls "
        "in 90s streetwear. Her auburn hair whips in the rooftop wind. The city skyline "
        "behind her — brutalist towers with holographic graffiti of sacred geometry. "
        "She raps with one hand gesturing precisely, the other on her chain. A DJ "
        "scratches on turntables behind her. Warm amber light from below catches her "
        "horns and her hair. She looks like she's flirting with the whole city."
    ),
    "15-subway-v4": (
        f"{CASSIYAH_CORE} She walks through a retro-futurist subway station with a "
        "slow confident strut, looking over her shoulder at the camera. Her wild red "
        "hair flows behind her. Brutalist concrete pillars with Arabic tilework. "
        "Holographic graffiti murals of topology diagrams on the tunnel walls. She "
        "wears a bomber jacket unzipped over a fitted crop top, gold chain swinging "
        "with her walk. A breakdancer spins on cardboard in the background. Green and "
        "amber fluorescent light. She's not just walking — she's arriving. 16mm grain."
    ),
    "16-nightclub-v4": (
        f"{CASSIYAH_CORE} She stands on a speaker stack in a dark underground nightclub, "
        "looking down at the crowd with a knowing smirk. Her fiery hair catches the "
        "laser light, glowing like actual flame. Geometric projections on the walls — "
        "category theory diagrams as light art. She holds the mic low, other hand on "
        "her hip. Her horns cast dramatic shadows in the strobe. Deep purple and "
        "electric gold strobes. She is the centre of gravity. Bass energy, 16mm grain."
    ),
}

flux = FluxGenerator(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="black-forest-labs/flux.2-max",
)

for key, prompt in TEST_SCENES.items():
    out = REFS_DIR / f"test_{key}.png"
    print(f"\n{'='*50}")
    print(f"  {key}")
    print(f"{'='*50}")
    try:
        flux.generate(prompt=prompt, output_path=out)
    except Exception as e:
        print(f"  FAILED: {e}")
    time.sleep(3)

print(f"\nDone. Check {REFS_DIR}/")
