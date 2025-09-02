from proposal_rag.services.document_processor import HDR_RE, BULLET_RE, PROMPTS_FILE

def main():
    print("PROMPTS_FILE =", PROMPTS_FILE)
    print("HDR_RE.match('# Заголовок'):", bool(HDR_RE.match("# Заголовок")))
    print("HDR_RE.match('Глава 2. Лечение'):", bool(HDR_RE.match("Глава 2. Лечение")))
    print("HDR_RE.match('Показатели:'):", bool(HDR_RE.match("Показатели:")))
    print("BULLET_RE.match('- пункт'):", bool(BULLET_RE.match("- пункт")))
    print("BULLET_RE.match('  1. нумерация'):", bool(BULLET_RE.match("  1. нумерация")))

if __name__ == "__main__":
    main()