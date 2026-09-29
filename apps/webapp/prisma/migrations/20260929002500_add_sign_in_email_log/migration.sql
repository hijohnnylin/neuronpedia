-- CreateTable
CREATE TABLE "SignInEmailLog" (
    "id" TEXT NOT NULL,
    "to" TEXT NOT NULL,
    "toKey" TEXT NOT NULL,
    "sentAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "SignInEmailLog_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "SignInEmailLog_toKey_sentAt_idx" ON "SignInEmailLog"("toKey", "sentAt");

-- CreateIndex
CREATE INDEX "SignInEmailLog_sentAt_idx" ON "SignInEmailLog"("sentAt");
