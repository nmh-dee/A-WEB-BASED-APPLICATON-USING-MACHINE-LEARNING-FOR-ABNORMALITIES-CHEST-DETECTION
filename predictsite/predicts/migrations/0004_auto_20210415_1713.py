# Generated by Django 3.1.7 on 2021-04-15 10:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predicts', '0003_auto_20210415_1712'),
    ]

    operations = [
        migrations.AlterField(
            model_name='predict',
            name='xray_predicted',
            field=models.ImageField(null=True, upload_to=''),
        ),
    ]
