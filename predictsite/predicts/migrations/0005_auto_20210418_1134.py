# Generated by Django 3.1.7 on 2021-04-18 04:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predicts', '0004_auto_20210415_1713'),
    ]

    operations = [
        migrations.AlterField(
            model_name='predict',
            name='xray',
            field=models.ImageField(null=True, upload_to='user/'),
        ),
        migrations.AlterField(
            model_name='predict',
            name='xray_predicted',
            field=models.ImageField(null=True, upload_to='truth/'),
        ),
        migrations.AlterField(
            model_name='predict',
            name='xray_truth',
            field=models.ImageField(null=True, upload_to='truth_xray/'),
        ),
    ]
