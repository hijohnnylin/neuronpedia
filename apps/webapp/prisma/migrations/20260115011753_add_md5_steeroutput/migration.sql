-- AlterTable
ALTER TABLE "SteerOutput" ADD COLUMN     "inputTextChatTemplateMd5" TEXT,
ADD COLUMN     "inputTextMd5" TEXT;
UPDATE "SteerOutput" SET "inputTextMd5" = MD5("inputText"), "inputTextChatTemplateMd5" = MD5("inputTextChatTemplate");