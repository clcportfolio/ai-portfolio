# Influencer Engagement Pipeline — Non-Technical Overview

## What It Does

This system predicts how well a social media influencer will perform in terms
of audience engagement. Given information about an influencer — their follower
count, posting habits, content type, and audience demographics — it classifies
them into **high**, **medium**, or **low** engagement tiers.

## Why It Matters

Influencer marketing platforms need to help brands find the right influencers
for their campaigns. Knowing which influencers are likely to generate strong
engagement saves brands money and improves campaign outcomes. This pipeline
automates that prediction and continuously monitors whether the model's
assumptions still hold as social media trends shift.

## What You See When You Run It

The dashboard has four views:

1. **Model Performance** — How accurate the predictions are, broken down by
   engagement tier. Includes a confusion matrix showing where the model
   gets it right and where it makes mistakes.

2. **Feature Importance** — Which factors matter most for predicting engagement.
   For example, you might see that follower count and posting frequency
   are the strongest predictors.

3. **Data Drift** — A monitoring view that flags when incoming data looks
   different from what the model was trained on. If influencer behavior
   patterns change significantly, this alerts the team to retrain the model.

4. **Pipeline Overview** — A diagram of how data flows through the system,
   from raw CSV files through processing, training, and monitoring.

## Who This Is For

Data teams at influencer marketing companies who need to automate engagement
prediction and maintain model quality over time.
