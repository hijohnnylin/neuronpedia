-- DropIndex
DROP INDEX "steerIndex";

-- DropIndex
DROP INDEX "steerIndex2";

-- DropIndex
DROP INDEX "steerIndexWithoutType";

-- DropIndex
DROP INDEX "steerIndexWithoutType2";

-- CreateIndex
CREATE INDEX "steerIndex" ON "SteerOutput"("modelId", "type", "inputTextMd5", "temperature", "numTokens", "freqPenalty", "seed", "strengthMultiplier", "version", "steerSpecialTokens", "steerMethod");

-- CreateIndex
CREATE INDEX "steerIndex2" ON "SteerOutput"("modelId", "type", "inputTextChatTemplateMd5", "temperature", "numTokens", "freqPenalty", "seed", "strengthMultiplier", "version", "steerSpecialTokens", "steerMethod");

-- CreateIndex
CREATE INDEX "steerIndexWithoutType" ON "SteerOutput"("modelId", "inputTextMd5", "temperature", "numTokens", "freqPenalty", "seed", "strengthMultiplier", "version", "steerSpecialTokens", "steerMethod");

-- CreateIndex
CREATE INDEX "steerIndexWithoutType2" ON "SteerOutput"("modelId", "inputTextChatTemplateMd5", "temperature", "numTokens", "freqPenalty", "seed", "strengthMultiplier", "version", "steerSpecialTokens", "steerMethod");
