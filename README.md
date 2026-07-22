# Sweet Tray Fill-Level Monitoring

A small computer vision project I built for a local sweet shop as a pilot — the idea was to see if a regular camera feed could tell you when a tray of sweets is running low, without someone having to walk the floor and check.

## The problem

In a sweet shop, trays get emptied out at different rates throughout the day. Staff usually catch this by eye, but that means someone has to notice, which doesn't always happen quickly. This project was about testing whether a camera + some CV could do that noticing automatically and flag trays that need a refill.

## How it works

There are two YOLOv8 models doing the heavy lifting:

- One finds and outlines each tray in the frame.
- The other finds the empty patches inside trays (the parts where there's no sweets left).

I track each tray across frames using ByteTrack so it keeps the same ID even if it moves slightly or gets briefly blocked from view. For every tray, I take its outline, shrink it in a bit so I'm not picking up rim/edge noise, and then check how much of that shrunk area overlaps with the "empty" mask. That overlap gives a rough 0–1 score for how empty the tray is.

That raw score jumps around a bit frame to frame, so I smooth it out over time (basically an exponential moving average) instead of trusting any single frame. On top of that I added a bit of hysteresis — meaning a tray has to clearly cross into a new state before its label actually changes, rather than flickering between "Half" and "Empty" every time the number wobbles near the line.

Once a tray's smoothed score crosses a threshold, it gets labeled:

- **Full** — plenty left
- **Half** — getting there
- **Empty** — triggers a "REFILL NEEDED" flag on screen

## What you actually see

The output is an annotated video with:

- Colored outlines around each tray showing its current status
- A red alert box + "REFILL NEEDED" text when a tray needs attention
- A small panel in the corner listing every tray's ID and fill %
- Running totals for how many trays are Full/Half/Empty at any given moment

There's also a debug mode that saves side-by-side crops (raw vs. annotated) for the first few frames — mostly so I could sanity-check that the masks were actually lining up with real trays and not just noise.

## Stack

- **Ultralytics YOLOv8** for both detection models 
- **OpenCV** for all the mask work, drawing, and video handling
- **ByteTrack** for keeping tray IDs consistent across frames
- **NumPy** for the ratio math

## A few notes on tuning

Everything from confidence thresholds to how aggressively the smoothing works is exposed as constants at the top of the script, since a lot of this pilot was trial and error — getting the empty-detection model to run less often than the tray model (since it's the heavier one), figuring out how much to erode the tray mask so rims didn't get counted as "empty," and adjusting the hysteresis margin so trays weren't flipping labels every few seconds.

## Where this stands

This was a pilot, not a finished product. It ran fine on CPU for testing, but it wasn't built with real-time deployment, multiple cameras, or live alerting in mind, that'd be the next step if the shop wanted to actually put this into daily use.

## Known rough edges

- Lighting changes and awkward camera angles can throw off the empty-region masks
- Long occlusions can occasionally cause a tray to lose its tracked ID and get treated as "new"
- Right now it's tuned for one specific camera setup, a different angle or tray layout would probably need re-tuning the thresholds
