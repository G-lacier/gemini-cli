/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import OpenAI from 'openai';
import {
  GenerateContentParameters,
  GenerateContentResponse,
  Part,
  Content,
  ContentListUnion,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
} from '@google/genai';
import {
  ContentGenerator,
  ContentGeneratorConfig,
} from './contentGenerator.js';

export class OpenAIContentGenerator implements ContentGenerator {
  private openai: OpenAI;
  private model: string;

  constructor(config: ContentGeneratorConfig) {
    this.openai = new OpenAI({ apiKey: config.apiKey });
    this.model = config.model || 'gpt-3.5-turbo';
  }

  private toArray(contents: ContentListUnion): Content[] {
    return Array.isArray(contents)
      ? (contents as Content[])
      : [contents as Content];
  }

  private convertMessages(
    contents: ContentListUnion,
  ): Array<{ role: 'assistant' | 'user'; content: string }> {
    return this.toArray(contents).map((c) => ({
      role: c.role === 'model' ? 'assistant' : 'user',
      content: (c.parts?.[0] as Part)?.text ?? '',
    }));
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const messages = this.convertMessages(request.contents);
    const resp = await this.openai.chat.completions.create({
      model: this.model,
      messages: messages as Parameters<
        OpenAI['chat']['completions']['create']
      >[0]['messages'],
    });
    return {
      candidates: [
        {
          content: {
            role: 'model',
            parts: [{ text: resp.choices[0].message.content || '' }],
          },
        },
      ],
    } as GenerateContentResponse;
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const messages = this.convertMessages(request.contents);
    const stream = await this.openai.chat.completions.create({
      model: this.model,
      messages: messages as Parameters<
        OpenAI['chat']['completions']['create']
      >[0]['messages'],
      stream: true,
    });
    async function* generator() {
      for await (const chunk of stream) {
        const text = chunk.choices[0]?.delta?.content;
        if (!text) continue;
        yield {
          candidates: [
            {
              content: { role: 'model', parts: [{ text }] },
            },
          ],
        } as GenerateContentResponse;
      }
    }
    return generator();
  }

  async countTokens(
    _request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    throw new Error('countTokens not implemented for OpenAI');
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error('embedContent not implemented for OpenAI');
  }
}
