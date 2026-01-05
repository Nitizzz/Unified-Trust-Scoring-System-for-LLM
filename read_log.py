try:
    with open('training.log', 'r', encoding='utf-16') as f:
        lines = f.readlines()
        print(''.join(lines[-30:]))
except Exception as e:
    print(e)
    # try utf-8
    try:
        with open('training.log', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(''.join(lines[-30:]))
    except Exception as e2:
        print(e2)
