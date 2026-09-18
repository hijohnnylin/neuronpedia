-- The steer lookup key includes presencePenalty in place of freqPenalty, which no backend applies.
--
-- This runs inside a transaction and takes a lock on "SteerOutput" for the whole build. That is
-- fine for a fresh or small database. On a large database, run prisma/manual/<this name>.sql by
-- hand instead (it builds CONCURRENTLY), then mark this migration applied:
--   npx prisma migrate resolve --applied 20260918200100_steeroutput_presence_penalty_indexes

-- DropIndex
DROP INDEX "steerIndex";

-- DropIndex
DROP INDEX "steerIndex2";

-- DropIndex
DROP INDEX "steerIndexWithoutType";

-- DropIndex
DROP INDEX "steerIndexWithoutType2";

-- CreateIndex
CREATE INDEX "steerIndex" ON "SteerOutput"("modelId", "type", "inputTextMd5", "temperature", "numTokens", "presencePenalty", "seed", "strengthMultiplier", "version", "steerSpecialTokens", "steerMethod");

-- CreateIndex
CREATE INDEX "steerIndex2" ON "SteerOutput"("modelId", "type", "inputTextChatTemplateMd5", "temperature", "numTokens", "presencePenalty", "seed", "strengthMultiplier", "version", "steerSpecialTokens", "steerMethod");

-- CreateIndex
CREATE INDEX "steerIndexWithoutType" ON "SteerOutput"("modelId", "inputTextMd5", "temperature", "numTokens", "presencePenalty", "seed", "strengthMultiplier", "version", "steerSpecialTokens", "steerMethod");

-- CreateIndex
CREATE INDEX "steerIndexWithoutType2" ON "SteerOutput"("modelId", "inputTextChatTemplateMd5", "temperature", "numTokens", "presencePenalty", "seed", "strengthMultiplier", "version", "steerSpecialTokens", "steerMethod");
