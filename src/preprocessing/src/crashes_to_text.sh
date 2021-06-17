#!/bin/bash
for filename in crash-*.pklz; do
  nipypecli crash "$filename" > "$(basename "$filename" .txt).txt"
done