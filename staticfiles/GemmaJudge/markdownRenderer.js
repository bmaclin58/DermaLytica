(function (global) {
    "use strict";

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function escapeAttribute(value) {
        return escapeHtml(value).replace(/`/g, "&#96;");
    }

    function sanitizeUrl(url) {
        var cleanUrl = String(url || "").trim();
        if (!cleanUrl) {
            return "";
        }

        if (/^(https?:\/\/|mailto:)/i.test(cleanUrl)) {
            return cleanUrl;
        }

        return "";
    }

    function parseInline(text) {
        var working = String(text || "");
        var tokens = [];

        function stash(html) {
            var token = "@@MDTOKEN" + tokens.length + "@@";
            tokens.push(html);
            return token;
        }

        working = working.replace(/`([^`]+)`/g, function (_match, code) {
            return stash("<code>" + escapeHtml(code) + "</code>");
        });

        working = working.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function (_match, label, url) {
            var safeUrl = sanitizeUrl(url);
            if (!safeUrl) {
                return escapeHtml(label);
            }

            return stash(
                '<a href="' + escapeAttribute(safeUrl) + '" target="_blank" rel="noopener noreferrer">' +
                escapeHtml(label) +
                "</a>"
            );
        });

        working = escapeHtml(working);
        working = working.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
        working = working.replace(/__([^_]+)__/g, "<strong>$1</strong>");
        working = working.replace(/\*([^*]+)\*/g, "<em>$1</em>");
        working = working.replace(/_([^_]+)_/g, "<em>$1</em>");

        return working.replace(/@@MDTOKEN(\d+)@@/g, function (_match, index) {
            return tokens[Number(index)] || "";
        });
    }

    function collectList(lines, startIndex, ordered) {
        var items = [];
        var index = startIndex;
        var markerPattern = ordered ? /^\d+\.\s+(.*)$/ : /^[-*+]\s+(.*)$/;

        while (index < lines.length) {
            var line = lines[index];
            var markerMatch = line.match(markerPattern);
            if (markerMatch) {
                items.push(markerMatch[1].trim());
                index += 1;
                continue;
            }

            if (/^\s{2,}\S/.test(line) && items.length) {
                items[items.length - 1] += " " + line.trim();
                index += 1;
                continue;
            }

            break;
        }

        return {
            html: "<" + (ordered ? "ol" : "ul") + ">" + items.map(function (item) {
                return "<li>" + parseInline(item) + "</li>";
            }).join("") + "</" + (ordered ? "ol" : "ul") + ">",
            nextIndex: index
        };
    }

    function render(markdown) {
        var lines = String(markdown || "").replace(/\r\n?/g, "\n").split("\n");
        var blocks = [];
        var index = 0;

        while (index < lines.length) {
            var line = lines[index];

            if (!line.trim()) {
                index += 1;
                continue;
            }

            if (/^```/.test(line)) {
                var language = line.slice(3).trim();
                var codeLines = [];
                index += 1;

                while (index < lines.length && !/^```/.test(lines[index])) {
                    codeLines.push(lines[index]);
                    index += 1;
                }

                if (index < lines.length && /^```/.test(lines[index])) {
                    index += 1;
                }

                blocks.push(
                    "<pre><code" +
                    (language ? ' class="language-' + escapeAttribute(language) + '"' : "") +
                    ">" +
                    escapeHtml(codeLines.join("\n")) +
                    "</code></pre>"
                );
                continue;
            }

            var headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
            if (headingMatch) {
                var level = headingMatch[1].length;
                blocks.push("<h" + level + ">" + parseInline(headingMatch[2].trim()) + "</h" + level + ">");
                index += 1;
                continue;
            }

            if (/^>\s?/.test(line)) {
                var quoteLines = [];
                while (index < lines.length && /^>\s?/.test(lines[index])) {
                    quoteLines.push(lines[index].replace(/^>\s?/, "").trim());
                    index += 1;
                }
                blocks.push("<blockquote><p>" + parseInline(quoteLines.join(" ")) + "</p></blockquote>");
                continue;
            }

            if (/^[-*+]\s+/.test(line)) {
                var unorderedList = collectList(lines, index, false);
                blocks.push(unorderedList.html);
                index = unorderedList.nextIndex;
                continue;
            }

            if (/^\d+\.\s+/.test(line)) {
                var orderedList = collectList(lines, index, true);
                blocks.push(orderedList.html);
                index = orderedList.nextIndex;
                continue;
            }

            var paragraphLines = [line.trim()];
            index += 1;
            while (index < lines.length) {
                var nextLine = lines[index];
                if (
                    !nextLine.trim() ||
                    /^```/.test(nextLine) ||
                    /^(#{1,6})\s+/.test(nextLine) ||
                    /^>\s?/.test(nextLine) ||
                    /^[-*+]\s+/.test(nextLine) ||
                    /^\d+\.\s+/.test(nextLine)
                ) {
                    break;
                }

                paragraphLines.push(nextLine.trim());
                index += 1;
            }

            blocks.push("<p>" + parseInline(paragraphLines.join(" ")) + "</p>");
        }

        if (!blocks.length) {
            return '<div class="markdown-body"><p></p></div>';
        }

        return '<div class="markdown-body">' + blocks.join("") + "</div>";
    }

    global.GemmaMarkdown = {
        escapeHtml: escapeHtml,
        render: render
    };
})(window);
